
import asyncio
import logging
import base64
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import litellm
from workers.config import Config
from workers.models import ModelEndpoint

logger = logging.getLogger(__name__)


@dataclass
class ImageReference:
    alt_text: str
    filename: str
    start_index: int
    end_index: int


class VisionService:
    """Describes figures with any vision model (Gemini, OpenAI, Claude, Ollama llava, ...) via LiteLLM."""

    def __init__(self, vision: ModelEndpoint):
        self.vision = vision
        self.context_window = Config.VISION_CONTEXT_WINDOW
        self.semaphore = asyncio.Semaphore(Config.LLM_MAX_CONCURRENT_CALLS)

        logger.info(f"VisionService initialized with model: {vision.model}")

    async def caption_images_with_context(
        self,
        markdown_content: str,
        images: List,  # List[ImageFile] from pdf_to_markdown
        image_refs: List  # List[MarkdownImage] from MarkdownChef
    ) -> Dict[str, str]:
        logger.info(f"Captioning {len(images)} images with context")

        ref_map = {}
        for ref in image_refs:
            filename = Path(ref.content).name
            ref_map[filename] = ref

        tasks = []
        for img in images:
            context = None
            if img.filename in ref_map:
                ref = ref_map[img.filename]
                context = self._extract_image_context(
                    markdown_content,
                    ref.start_index,
                    ref.end_index
                )

            task = self._caption_single_image(
                img.path,
                context=context,
                filename=img.filename
            )
            tasks.append(task)

        captions_list = await asyncio.gather(*tasks)

        captions = {
            img.filename: caption
            for img, caption in zip(images, captions_list)
        }

        logger.info(f"Image captioning complete: {len(captions)} captions")

        return captions

    def _extract_image_context(
        self,
        markdown: str,
        img_start: int,
        img_end: int
    ) -> str:

        window = self.context_window

        context_start = max(0, img_start - window)
        before = markdown[context_start:img_start]

        context_end = min(len(markdown), img_end + window)
        after = markdown[img_end:context_end]

        context = f"{before.strip()}\n[IMAGE POSITION]\n{after.strip()}"

        logger.debug(f"Extracted context ({len(context)} chars) for image at {img_start}")

        return context

    @staticmethod
    def _build_prompt(context: Optional[str]) -> str:
        if context:
            return f"""
Context from the academic paper:
{context}

Describe this figure in detail:
1. Type of visualization (chart/diagram/graph/equation/screenshot/etc)
2. What is being measured or shown (axes, labels, units)
3. Key data points, trends, or elements visible
4. How it relates to the surrounding context
5. Main conclusion or insight supported by this figure

Be specific and include all visible text, numbers, and labels.
"""
        return """
Describe this academic figure in detail:
1. Type of visualization (chart/diagram/graph/equation/screenshot/etc)
2. Axes/labels/legends visible (include all text)
3. Key data points or trends
4. Main conclusion supported

Be specific and include all visible text and numbers.
"""

    async def _caption_single_image(
        self,
        image_path: Path,
        context: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        async with self.semaphore:
            try:
                mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
                img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
                response = await litellm.acompletion(
                    model=self.vision.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._build_prompt(context)},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                        ],
                    }],
                    max_tokens=800,
                    **self.vision.call_kwargs(),
                )
                caption = (response.choices[0].message.content or "").strip()

                logger.info(f"Captioned {filename}: {len(caption)} chars")
                return caption

            except Exception as e:
                logger.error(f"Failed to caption {filename}: {str(e)}")
                return f"[Image: {filename} - caption failed: {str(e)}]"
