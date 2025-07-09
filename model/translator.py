"""
Translation Module

This module provides Vietnamese translation functionality using deep-translator library.
"""

from deep_translator import GoogleTranslator, MyMemoryTranslator, LingueeTranslator
import logging
from typing import List, Optional
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietnameseTranslator:
    """Vietnamese translator using multiple translation services as fallback"""
    
    def __init__(self, primary_service: str = "google", max_retries: int = 3):
        """
        Initialize Vietnamese translator
        
        Args:
            primary_service: Primary translation service ('google', 'mymemory', 'linguee')
            max_retries: Maximum number of retry attempts
        """
        self.primary_service = primary_service
        self.max_retries = max_retries
        self.translators = self._init_translators()
        
    def _init_translators(self) -> dict:
        """Initialize different translation services"""
        translators = {}
        
        try:
            translators['google'] = GoogleTranslator(source='en', target='vi')
            logger.info("Google Translator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Google Translator: {e}")
        
        try:
            translators['mymemory'] = MyMemoryTranslator(source='en', target='vi')
            logger.info("MyMemory Translator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MyMemory Translator: {e}")
            
        try:
            translators['linguee'] = LingueeTranslator(source='en', target='vi')
            logger.info("Linguee Translator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Linguee Translator: {e}")
        
        return translators
    
    def translate(self, text: str, service: Optional[str] = None) -> str:
        """
        Translate English text to Vietnamese
        
        Args:
            text: English text to translate
            service: Specific service to use (optional)
            
        Returns:
            Vietnamese translation
        """
        if not text or not text.strip():
            return ""
        
        # Clean input text
        text = self._clean_text(text)
        
        # Determine which service to use
        service_to_use = service or self.primary_service
        
        # Try primary service first
        if service_to_use in self.translators:
            result = self._translate_with_service(text, service_to_use)
            if result:
                return self._post_process_translation(result)
        
        # Try fallback services if primary fails
        for service_name, translator in self.translators.items():
            if service_name != service_to_use:
                logger.info(f"Trying fallback service: {service_name}")
                result = self._translate_with_service(text, service_name)
                if result:
                    return self._post_process_translation(result)
        
        # If all services fail, return original text
        logger.error("All translation services failed")
        return text
    
    def _translate_with_service(self, text: str, service: str) -> Optional[str]:
        """
        Translate text using specific service with retry logic
        
        Args:
            text: Text to translate
            service: Service name to use
            
        Returns:
            Translated text or None if failed
        """
        translator = self.translators.get(service)
        if not translator:
            return None
        
        for attempt in range(self.max_retries):
            try:
                result = translator.translate(text)
                if result and result.strip():
                    logger.info(f"Successfully translated using {service}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed with {service}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Wait before retry
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean input text before translation"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Ensure proper capitalization
        text = text[0].upper() + text[1:] if text else text
        
        # Add period if missing and text doesn't end with punctuation
        if text and not text[-1] in '.!?':
            text += '.'
            
        return text
    
    def _post_process_translation(self, text: str) -> str:
        """Post-process translated text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Ensure proper capitalization
        if text:
            text = text[0].upper() + text[1:]
        
        # Fix common Vietnamese translation issues
        text = self._fix_vietnamese_issues(text)
        
        return text
    
    def _fix_vietnamese_issues(self, text: str) -> str:
        """Fix common issues in Vietnamese translations"""
        # Common fixes for Vietnamese text
        fixes = {
            'một người phụ nữ': 'một phụ nữ',
            'một người đàn ông': 'một người đàn ông',
            'đang ngồi trên': 'đang ngồi',
            'màu xanh da trời': 'màu xanh',
            'màu xanh lá cây': 'màu xanh lá',
        }
        
        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def translate_batch(self, texts: List[str], service: Optional[str] = None) -> List[str]:
        """
        Translate multiple texts
        
        Args:
            texts: List of English texts to translate
            service: Specific service to use (optional)
            
        Returns:
            List of Vietnamese translations
        """
        translations = []
        for text in texts:
            translation = self.translate(text, service)
            translations.append(translation)
            time.sleep(0.1)  # Small delay to avoid rate limiting
        
        return translations
    
    def get_available_services(self) -> List[str]:
        """Get list of available translation services"""
        return list(self.translators.keys())
    
    def test_services(self) -> dict:
        """Test all available translation services"""
        test_text = "A woman is sitting on the beach with her dog."
        results = {}
        
        for service in self.get_available_services():
            try:
                translation = self.translate(test_text, service)
                results[service] = {
                    'status': 'success',
                    'translation': translation
                }
            except Exception as e:
                results[service] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results


def create_translator(service: str = "google") -> VietnameseTranslator:
    """
    Factory function to create a Vietnamese translator
    
    Args:
        service: Primary translation service to use
        
    Returns:
        VietnameseTranslator instance
    """
    return VietnameseTranslator(primary_service=service) 