"""
Normalization Service - Converts raw documents to structured format.

Handles:
- PDF text extraction (with OCR fallback)
- HTML content extraction and boilerplate removal
- CSV/Excel table parsing
- JSON structure flattening
- Section detection and hierarchy
- Table extraction
- Language detection
"""
import io
import logging
import re
from typing import Optional
from uuid import uuid4

import pandas as pd
import pdfplumber
import pytesseract
from bs4 import BeautifulSoup
from langdetect import detect
from PIL import Image
from pypdf import PdfReader
from trafilatura import extract

from shared.config.settings import get_settings
from shared.database.mongodb import get_mongodb
from shared.models.schemas import (
    DocumentType,
    NormalizedDocument,
    Section,
    Table,
)

logger = logging.getLogger(__name__)


class NormalizationService:
    """Service for normalizing raw documents into structured format."""
    
    def __init__(self):
        self.settings = get_settings()
        self.mongodb = get_mongodb()
        self.raw_docs = self.mongodb.get_async_collection("raw_documents")
        self.normalized_docs = self.mongodb.get_async_collection("normalized_docs")
        
    async def normalize_document(
        self,
        document_id: str,
    ) -> NormalizedDocument:
        """
        Normalize a raw document.
        
        Args:
            document_id: Raw document ID
            
        Returns:
            NormalizedDocument
        """
        logger.info(f"Normalizing document: {document_id}")
        
        # Get raw document
        raw_doc = await self.raw_docs.find_one({"document_id": document_id})
        if not raw_doc:
            raise ValueError(f"Document not found: {document_id}")
        
        # Get content
        from services.ingestion.service import IngestionService
        ingestion = IngestionService()
        content = await ingestion.get_document_content(document_id)
        
        # Route to appropriate normalizer
        source_type = DocumentType(raw_doc["source_type"])
        
        if source_type == DocumentType.PDF:
            normalized = await self._normalize_pdf(document_id, content, raw_doc)
        elif source_type == DocumentType.HTML:
            normalized = await self._normalize_html(document_id, content, raw_doc)
        elif source_type == DocumentType.CSV:
            normalized = await self._normalize_csv(document_id, content, raw_doc)
        elif source_type == DocumentType.JSON or source_type == DocumentType.API:
            normalized = await self._normalize_json(document_id, content, raw_doc)
        elif source_type == DocumentType.TEXT:
            normalized = await self._normalize_text(document_id, content, raw_doc)
        else:
            raise ValueError(f"Unsupported document type: {source_type}")
        
        # Detect language
        if normalized.text:
            try:
                normalized.language = detect(normalized.text[:1000])
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
                normalized.language = "unknown"
        
        # Insert into MongoDB
        await self.normalized_docs.insert_one(normalized.model_dump())
        
        logger.info(
            f"Document normalized: {normalized.document_id} "
            f"(sections: {len(normalized.sections)}, "
            f"tables: {len(normalized.tables)}, "
            f"words: {normalized.word_count})"
        )
        
        # Emit extraction task
        await self._emit_extraction_task(normalized.document_id)
        
        return normalized
    
    async def _normalize_pdf(
        self,
        document_id: str,
        content: bytes,
        raw_doc: dict,
    ) -> NormalizedDocument:
        """Normalize PDF document."""
        logger.debug(f"Normalizing PDF: {document_id}")
        
        text_parts = []
        sections = []
        tables = []
        char_offset = 0
        
        try:
            # Try pdfplumber first (better for tables)
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    page_text = page.extract_text() or ""
                    if page_text:
                        section = Section(
                            section_id=f"{document_id}_page_{page_num}",
                            heading=f"Page {page_num}",
                            level=1,
                            text=page_text,
                            start_char=char_offset,
                            end_char=char_offset + len(page_text),
                            metadata={"page_number": page_num},
                        )
                        sections.append(section)
                        text_parts.append(page_text)
                        char_offset += len(page_text) + 1
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table_idx, table_data in enumerate(page_tables):
                            if table_data and len(table_data) > 1:
                                headers = table_data[0]
                                rows = table_data[1:]
                                
                                table = Table(
                                    table_id=f"{document_id}_table_{page_num}_{table_idx}",
                                    headers=[str(h) for h in headers if h],
                                    rows=[[str(cell) for cell in row] for row in rows],
                                    page_number=page_num,
                                )
                                tables.append(table)
        
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying pypdf: {e}")
            
            # Fallback to pypdf
            try:
                reader = PdfReader(io.BytesIO(content))
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text() or ""
                    if page_text:
                        section = Section(
                            section_id=f"{document_id}_page_{page_num}",
                            heading=f"Page {page_num}",
                            level=1,
                            text=page_text,
                            start_char=char_offset,
                            end_char=char_offset + len(page_text),
                            metadata={"page_number": page_num},
                        )
                        sections.append(section)
                        text_parts.append(page_text)
                        char_offset += len(page_text) + 1
            except Exception as e2:
                logger.error(f"pypdf also failed, attempting OCR: {e2}")
                
                # Last resort: OCR
                try:
                    text_parts = await self._ocr_pdf(content)
                except Exception as e3:
                    logger.error(f"OCR failed: {e3}")
                    text_parts = ["[PDF extraction failed]"]
        
        full_text = "\n".join(text_parts)
        
        return NormalizedDocument(
            document_id=f"norm_{uuid4().hex[:12]}",
            raw_document_id=document_id,
            text=full_text,
            sections=sections,
            tables=tables,
            word_count=len(full_text.split()),
            char_count=len(full_text),
        )
    
    async def _normalize_html(
        self,
        document_id: str,
        content: bytes,
        raw_doc: dict,
    ) -> NormalizedDocument:
        """Normalize HTML document."""
        logger.debug(f"Normalizing HTML: {document_id}")
        
        html_str = content.decode("utf-8", errors="ignore")
        
        # Use trafilatura for main content extraction (removes boilerplate)
        main_text = extract(html_str, include_tables=True)
        
        if not main_text:
            # Fallback to BeautifulSoup
            soup = BeautifulSoup(html_str, "lxml")
            
            # Remove scripts, styles
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            main_text = soup.get_text(separator="\n", strip=True)
        
        # Extract sections from headings
        sections = []
        soup = BeautifulSoup(html_str, "lxml")
        char_offset = 0
        
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(heading.name[1])
            heading_text = heading.get_text(strip=True)
            
            # Get text until next heading
            section_text_parts = []
            for sibling in heading.next_siblings:
                if sibling.name and sibling.name.startswith("h"):
                    break
                if hasattr(sibling, "get_text"):
                    section_text_parts.append(sibling.get_text(strip=True))
            
            section_text = "\n".join(section_text_parts)
            
            if section_text:
                section = Section(
                    section_id=f"{document_id}_section_{len(sections)}",
                    heading=heading_text,
                    level=level,
                    text=section_text,
                    start_char=char_offset,
                    end_char=char_offset + len(section_text),
                )
                sections.append(section)
                char_offset += len(section_text) + 1
        
        # Extract tables
        tables = []
        for table_idx, html_table in enumerate(soup.find_all("table")):
            try:
                # Parse table with pandas
                df = pd.read_html(str(html_table))[0]
                
                table = Table(
                    table_id=f"{document_id}_table_{table_idx}",
                    caption=html_table.find("caption").get_text(strip=True)
                    if html_table.find("caption")
                    else None,
                    headers=df.columns.tolist(),
                    rows=df.values.tolist(),
                )
                tables.append(table)
            except Exception as e:
                logger.warning(f"Failed to parse HTML table {table_idx}: {e}")
        
        return NormalizedDocument(
            document_id=f"norm_{uuid4().hex[:12]}",
            raw_document_id=document_id,
            text=main_text,
            sections=sections,
            tables=tables,
            word_count=len(main_text.split()),
            char_count=len(main_text),
        )
    
    async def _normalize_csv(
        self,
        document_id: str,
        content: bytes,
        raw_doc: dict,
    ) -> NormalizedDocument:
        """Normalize CSV document."""
        logger.debug(f"Normalizing CSV: {document_id}")
        
        # Parse CSV
        df = pd.read_csv(io.BytesIO(content))
        
        # Convert to table
        table = Table(
            table_id=f"{document_id}_table_0",
            headers=df.columns.tolist(),
            rows=df.values.tolist(),
        )
        
        # Generate text summary
        text = f"CSV Table with {len(df)} rows and {len(df.columns)} columns\n"
        text += f"Columns: {', '.join(df.columns.tolist())}\n"
        text += f"\nFirst few rows:\n{df.head(5).to_string()}"
        
        return NormalizedDocument(
            document_id=f"norm_{uuid4().hex[:12]}",
            raw_document_id=document_id,
            text=text,
            sections=[],
            tables=[table],
            word_count=len(text.split()),
            char_count=len(text),
        )
    
    async def _normalize_json(
        self,
        document_id: str,
        content: bytes,
        raw_doc: dict,
    ) -> NormalizedDocument:
        """Normalize JSON document."""
        logger.debug(f"Normalizing JSON: {document_id}")
        
        import json
        
        data = json.loads(content)
        
        # Flatten JSON to text
        def flatten_json(obj, prefix=""):
            lines = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (dict, list)):
                        lines.extend(flatten_json(value, new_prefix))
                    else:
                        lines.append(f"{new_prefix}: {value}")
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    new_prefix = f"{prefix}[{idx}]"
                    if isinstance(item, (dict, list)):
                        lines.extend(flatten_json(item, new_prefix))
                    else:
                        lines.append(f"{new_prefix}: {item}")
            return lines
        
        text_lines = flatten_json(data)
        text = "\n".join(text_lines)
        
        return NormalizedDocument(
            document_id=f"norm_{uuid4().hex[:12]}",
            raw_document_id=document_id,
            text=text,
            sections=[],
            tables=[],
            word_count=len(text.split()),
            char_count=len(text),
        )
    
    async def _normalize_text(
        self,
        document_id: str,
        content: bytes,
        raw_doc: dict,
    ) -> NormalizedDocument:
        """Normalize plain text document."""
        logger.debug(f"Normalizing text: {document_id}")
        
        text = content.decode("utf-8", errors="ignore")
        
        # Simple section detection based on double newlines
        paragraphs = re.split(r"\n\n+", text)
        sections = []
        char_offset = 0
        
        for idx, para in enumerate(paragraphs):
            if para.strip():
                section = Section(
                    section_id=f"{document_id}_para_{idx}",
                    level=1,
                    text=para,
                    start_char=char_offset,
                    end_char=char_offset + len(para),
                )
                sections.append(section)
                char_offset += len(para) + 2
        
        return NormalizedDocument(
            document_id=f"norm_{uuid4().hex[:12]}",
            raw_document_id=document_id,
            text=text,
            sections=sections,
            tables=[],
            word_count=len(text.split()),
            char_count=len(text),
        )
    
    async def _ocr_pdf(self, content: bytes) -> list[str]:
        """OCR fallback for PDF."""
        logger.info("Performing OCR on PDF")
        
        try:
            from pdf2image import convert_from_bytes
            
            images = convert_from_bytes(content)
            text_parts = []
            
            for idx, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                text_parts.append(text)
                logger.debug(f"OCR'd page {idx + 1}")
            
            return text_parts
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []
    
    async def _emit_extraction_task(self, document_id: str):
        """Emit extraction task."""
        try:
            from workers.tasks import extract_triples
            
            extract_triples.delay(document_id)
            logger.info(f"Extraction task emitted for: {document_id}")
        except Exception as e:
            logger.error(f"Failed to emit extraction task: {e}")
