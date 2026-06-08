
import asyncio
import logging
import base64
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from workers.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ImageReference:
    alt_text: str
    filename: str
    start_index: int
    end_index: int


class VisionService:

    def __init__(self, provider: Optional[str] = None, api_key: str = None, model_name: str = None):
        self.provider = provider or Config.VISION_MODEL_PROVIDER
        self.context_window = Config.VISION_CONTEXT_WINDOW
        self.semaphore = asyncio.Semaphore(Config.LLM_MAX_CONCURRENT_CALLS)

        if self.provider == "gemini":
            import google.generativeai as genai
            resolved_key = api_key or Config.GEMINI_API_KEY
            resolved_model = model_name or Config.VISION_MODEL_NAME
            genai.configure(api_key=resolved_key)
            self.model = genai.GenerativeModel(resolved_model)
        elif self.provider == "openai":
            import openai
            resolved_key = api_key or Config.OPENAI_API_KEY
            self.client = openai.AsyncOpenAI(api_key=resolved_key)
        else:
            raise ValueError(f"Unsupported vision provider: {self.provider}")

        logger.info(f"VisionService initialized with provider: {self.provider}")
    
    async def caption_images_with_context(
        self,
        markdown_content: str,
        images: List,  # List[ImageFile] from pdf_to_markdown
        image_refs: List  # List[MarkdownImage] from MarkdownChef
    ) -> Dict[str, str]:
        logger.info(f"Captioning {len(images)} images with context")
 
        image_map = {img.filename: img.path for img in images}
        
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
    
    async def _caption_single_image(
        self,
        image_path: Path,
        context: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        async with self.semaphore:
            try:
                if self.provider == "gemini":
                    caption = await self._caption_gemini(image_path, context)
                elif self.provider == "openai":
                    caption = await self._caption_openai(image_path, context)
                # elif self.provider == "ollama":
                #     caption = await self._caption_ollama(image_path, context)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                
                logger.info(f"Captioned {filename}: {len(caption)} chars")
                return caption
                
            except Exception as e:
                logger.error(f"Failed to caption {filename}: {str(e)}")
                return f"[Image: {filename} - caption failed: {str(e)}]"
    
    async def _caption_gemini(
        self,
        image_path: Path,
        context: Optional[str]
    ) -> str:
        """Caption image using Gemini vision model"""
        from PIL import Image
        
        # Build prompt
        if context:
            prompt = f"""
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
        else:
            prompt = """
Describe this academic figure in detail:
1. Type of visualization (chart/diagram/graph/equation/screenshot/etc)
2. Axes/labels/legends visible (include all text)
3. Key data points or trends
4. Main conclusion supported

Be specific and include all visible text and numbers.
"""
        
        # Load image
        img = Image.open(image_path)
        
        # Generate caption (synchronous Gemini API)
        response = await asyncio.to_thread(
            self.model.generate_content,
            [prompt, img]
        )
        
        return response.text.strip()
    
    async def _caption_openai(
        self,
        image_path: Path,
        context: Optional[str]
    ) -> str:
        """Caption image using OpenAI vision model"""
        
        # Encode image to base64
        with open(image_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Build prompt
        if context:
            prompt = f"""
Context from the academic paper:
{context}

Describe this figure in detail and how it relates to the context.
Include all visible text, data points, and insights.
"""
        else:
            prompt = """
Describe this academic figure in detail.
Include type, axes/labels, data points, and main conclusions.
Include all visible text.
"""

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }],
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    # async def _caption_ollama(
    #     self,
    #     image_path: Path,
    #     context: Optional[str]
    # ) -> str:
    #     """Caption image using Ollama (local) vision model"""
        
    #     # Build prompt
    #     if context:
    #         prompt = f"Context: {context}\n\nDescribe this figure and how it relates to the context."
    #     else:
    #         prompt = "Describe this academic figure in detail."
        
    #     # Call Ollama (synchronous)
    #     response = await asyncio.to_thread(
    #         self.ollama_client.chat,
    #         model='llava',
    #         messages=[{
    #             'role': 'user',
    #             'content': prompt,
    #             'images': [str(image_path)]
    #         }]
    #     )
        
    #     return response['message']['content'].strip()
