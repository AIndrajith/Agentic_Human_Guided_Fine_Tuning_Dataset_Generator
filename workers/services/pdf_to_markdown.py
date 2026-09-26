"""
Academic PDF -> Markdown with Marker 2.x, run as a child process (see marker_runner.py).

- Marker's LLM step (--use_llm) uses the project's VISION model when Marker supports that provider
  (Gemini, OpenAI, Claude, Azure, Ollama, OpenRouter); otherwise GEMINI_API_KEY from .env; otherwise no LLM.
- Keys travel in the child's environment, never on its command line.
- No --force_ocr by default: Marker 2 re-OCRs only pages/blocks whose text is bad (forcing it makes its
  local model re-read every page, which is very slow on CPU).
- Runs without blocking the worker's event loop; on timeout the whole process tree is killed.
"""
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from workers.config import Config
from workers.models import ModelEndpoint

logger = logging.getLogger(__name__)

RUNNER = Path(__file__).with_name("marker_runner.py")
# keys Marker (or google-genai) would pick up from the environment; the child only gets the ones we set
_KEY_ENV_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")


@dataclass
class ImageFile:
    filename: str
    path: Path


@dataclass
class MarkdownOutput:
    markdown_text: str
    markdown_path: Path
    images: List[ImageFile]
    metadata: Dict[str, Any]


def marker_llm_settings(vision: Optional[ModelEndpoint]) -> Tuple[Optional[str], Dict[str, Any], Dict[str, str]]:
    """(Marker service class, its config options, extra env vars) for the project's vision model.
    Service None = run Marker without an LLM."""
    if vision is not None:
        provider, _, model = vision.model.partition("/")
        key, base = vision.api_key, vision.api_base
        match provider:
            case "gemini":
                # GOOGLE_API_KEY too: Marker's settings let that env var override the configured key
                return ("marker.services.gemini.GoogleGeminiService",
                        {"gemini_api_key": key, "gemini_model_name": model}, {"GOOGLE_API_KEY": key or ""})
            case "openai":
                options = {"openai_api_key": key, "openai_model": model}
                if base:
                    options["openai_base_url"] = base.rstrip("/")
                return "marker.services.openai.OpenAIService", options, {}
            case "anthropic":
                return ("marker.services.claude.ClaudeService",
                        {"claude_api_key": key, "claude_model_name": model}, {})
            case "azure":
                return ("marker.services.azure_openai.AzureOpenAIService",
                        {"azure_endpoint": base, "azure_api_key": key,
                         "azure_api_version": vision.api_version, "deployment_name": model}, {})
            case "ollama" | "ollama_chat":
                return ("marker.services.ollama.OllamaService",
                        {"ollama_base_url": (base or "http://127.0.0.1:11434").rstrip("/"), "ollama_model": model}, {})
            case "openrouter":
                return ("marker.services.openrouter.OpenRouterService",
                        {"openrouter_api_key": key, "openrouter_model": model}, {})
        logger.warning(f"Marker has no service for vision provider '{provider}'; trying GEMINI_API_KEY from .env")

    if Config.GEMINI_API_KEY:
        return ("marker.services.gemini.GoogleGeminiService",
                {"gemini_api_key": Config.GEMINI_API_KEY}, {"GOOGLE_API_KEY": Config.GEMINI_API_KEY})
    logger.warning("No LLM available for Marker; converting without --use_llm (lower quality tables/math)")
    return None, {}, {}


class PDFToMarkdownService:

    def __init__(self):
        self.output_format = Config.MARKER_OUTPUT_FORMAT
        self.use_llm = Config.MARKER_USE_LLM
        self.force_ocr = Config.MARKER_FORCE_OCR
        self.redo_inline_math = Config.MARKER_REDO_INLINE_MATH
        self.mode = Config.MARKER_MODE
        self.timeout = Config.MARKER_TIMEOUT

        logger.info(
            f"PDFToMarkdownService initialized: "
            f"format={self.output_format}, use_llm={self.use_llm}, force_ocr={self.force_ocr}, "
            f"redo_math={self.redo_inline_math}, mode={self.mode or 'auto'}, timeout={self.timeout}s"
        )

    def build_run(self, pdf_path: Path, output_dir: Path, vision: Optional[ModelEndpoint]) -> Tuple[dict, dict]:
        """(runner config, extra env vars). Kept separate so it can be tested without running Marker."""
        config: Dict[str, Any] = {
            "pdf_path": str(pdf_path),
            "output_dir": str(output_dir),
            "output_format": self.output_format,
            "force_ocr": self.force_ocr,
        }
        if self.mode:
            config["mode"] = self.mode

        env: Dict[str, str] = {}
        service, options, service_env = marker_llm_settings(vision) if self.use_llm else (None, {}, {})
        if service:
            config.update({"use_llm": True, "llm_service": service, "redo_inline_math": self.redo_inline_math})
            config.update({k: v for k, v in options.items() if v})
            env.update(service_env)
        return config, env

    async def convert_pdf(
        self, pdf_path: Path, output_dir: Path, vision: Optional[ModelEndpoint] = None
    ) -> MarkdownOutput:

        logger.info(f"Converting PDF to markdown: {pdf_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        config, extra_env = self.build_run(pdf_path, output_dir, vision)
        await self._run_marker(config, extra_env)

        # Parse Marker output
        markdown_path = self._find_markdown_output(output_dir, pdf_path)
        markdown_text = markdown_path.read_text(encoding='utf-8')

        images = self._extract_images(output_dir)

        metadata = {
            "source_pdf": str(pdf_path),
            "markdown_path": str(markdown_path),
            "image_count": len(images),
            "marker_config": {
                "use_llm": bool(config.get("use_llm")),
                "llm_service": config.get("llm_service"),
                "force_ocr": self.force_ocr,
                "redo_inline_math": bool(config.get("redo_inline_math")),
                "mode": self.mode or "auto",
            }
        }

        logger.info(
            f"PDF conversion complete: {len(markdown_text)} chars, "
            f"{len(images)} images"
        )

        return MarkdownOutput(
            markdown_text=markdown_text,
            markdown_path=markdown_path,
            images=images,
            metadata=metadata
        )

    async def _run_marker(self, config: dict, extra_env: dict) -> None:
        env = {k: v for k, v in os.environ.items() if k not in _KEY_ENV_VARS}
        env.update(extra_env)
        env["MARKER_RUNNER_CONFIG"] = json.dumps(config)

        logger.info(f"Running Marker (llm_service={config.get('llm_service')}, timeout={self.timeout}s)")
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(RUNNER),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            **({"start_new_session": True} if os.name != "nt" else {}),
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            self._kill_tree(process.pid)
            await process.wait()
            raise TimeoutError(f"Marker conversion timed out after {self.timeout}s")

        if stdout:
            logger.debug(f"Marker output: {stdout.decode(errors='replace')[-2000:]}")
        if process.returncode != 0:
            tail = stderr.decode(errors="replace")[-1500:] if stderr else ""
            raise RuntimeError(f"Marker failed (exit code {process.returncode}): {tail}")

    @staticmethod
    def _kill_tree(pid: int) -> None:
        """Kill Marker and anything it started (e.g. its local inference server)."""
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
            else:
                os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError) as e:
            logger.warning(f"Could not kill Marker process tree {pid}: {e}")

    def _find_markdown_output(self, output_dir: Path, pdf_path: Path) -> Path:

        pdf_stem = pdf_path.stem
        markdown_path = output_dir / f"{pdf_stem}.md"

        if markdown_path.exists():
            return markdown_path

        markdown_path = output_dir / pdf_stem / f"{pdf_stem}.md"

        if markdown_path.exists():
            return markdown_path

        md_files = list(output_dir.rglob("*.md"))

        if md_files:
            logger.warning(
                f"Using first found markdown file: {md_files[0]}"
            )
            return md_files[0]

        raise FileNotFoundError(
            f"No markdown output found in {output_dir}"
        )

    def _extract_images(self, output_dir: Path) -> List[ImageFile]:

        images = []

        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.svg'}

        for img_path in output_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                images.append(ImageFile(
                    filename=img_path.name,
                    path=img_path
                ))

        logger.info(f"Found {len(images)} extracted images")

        return images
