#!/usr/bin/env python3
"""
Internationalization (i18n) module for GhostGPU.

Provides multi-language support with JSON-based translation files.
Supports: English, Japanese, Chinese (Simplified), Korean, Spanish, French, German, Portuguese, Russian
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


class I18nManager:
    """Manager for multi-language support with translation fallback."""

    SUPPORTED_LANGUAGES = ['en', 'ja', 'zh', 'ko', 'es', 'fr', 'de', 'pt', 'ru']

    def __init__(self, locale_dir: str = "locale", default_language: str = "en"):
        """
        Initialize I18n manager.

        Parameters
        ----------
        locale_dir : str
            Directory containing translation JSON files
        default_language : str
            Default language code (e.g., 'en', 'ja')
        """
        self.locale_dir = Path(locale_dir)
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, Any]] = {}
        self._load_all_translations()

    def _load_all_translations(self) -> None:
        """Load all translation files from locale directory."""
        if not self.locale_dir.exists():
            self.locale_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_translations()
            return

        # Load all JSON files from locale directory
        for json_file in self.locale_dir.glob("*.json"):
            lang_code = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load {json_file}: {e}")

        # Ensure default language is loaded
        if self.default_language not in self.translations:
            self._create_default_translations()

    def _create_default_translations(self) -> None:
        """Create default translation files for all supported languages."""
        languages = {
            'en': {
                'app.name': 'GhostGPU',
                'app.description': 'A lightweight numerical computing runtime',
                'nav.brand': 'GhostGPU',
                'settings.title': 'Settings',
                'settings.language': 'Language',
                'settings.performance': 'Performance',
                'settings.memory': 'Memory Management',
                'error.not_found': 'Not found',
                'error.invalid_input': 'Invalid input: {input}',
            },
            'ja': {
                'app.name': 'GhostGPU',
                'app.description': '軽量な数値計算ランタイム',
                'nav.brand': 'GhostGPU',
                'settings.title': '設定',
                'settings.language': '言語',
                'settings.performance': 'パフォーマンス',
                'settings.memory': 'メモリ管理',
                'error.not_found': '見つかりません',
                'error.invalid_input': '無効な入力: {input}',
            },
            'zh': {
                'app.name': 'GhostGPU',
                'app.description': '轻量级数值计算运行时',
                'nav.brand': 'GhostGPU',
                'settings.title': '设置',
                'settings.language': '语言',
                'settings.performance': '性能',
                'settings.memory': '内存管理',
                'error.not_found': '未找到',
                'error.invalid_input': '无效输入: {input}',
            },
            'ko': {
                'app.name': 'GhostGPU',
                'app.description': '경량 수치 계산 런타임',
                'nav.brand': 'GhostGPU',
                'settings.title': '설정',
                'settings.language': '언어',
                'settings.performance': '성능',
                'settings.memory': '메모리 관리',
                'error.not_found': '찾을 수 없음',
                'error.invalid_input': '잘못된 입력: {input}',
            },
            'es': {
                'app.name': 'GhostGPU',
                'app.description': 'Un runtime de computación numérica ligero',
                'nav.brand': 'GhostGPU',
                'settings.title': 'Configuración',
                'settings.language': 'Idioma',
                'settings.performance': 'Rendimiento',
                'settings.memory': 'Gestión de memoria',
                'error.not_found': 'No encontrado',
                'error.invalid_input': 'Entrada inválida: {input}',
            },
            'fr': {
                'app.name': 'GhostGPU',
                'app.description': 'Un runtime léger de calcul numérique',
                'nav.brand': 'GhostGPU',
                'settings.title': 'Paramètres',
                'settings.language': 'Langue',
                'settings.performance': 'Performance',
                'settings.memory': 'Gestion de la mémoire',
                'error.not_found': 'Non trouvé',
                'error.invalid_input': 'Entrée invalide: {input}',
            },
            'de': {
                'app.name': 'GhostGPU',
                'app.description': 'Eine leichte numerische Berechnungslaufzeit',
                'nav.brand': 'GhostGPU',
                'settings.title': 'Einstellungen',
                'settings.language': 'Sprache',
                'settings.performance': 'Leistung',
                'settings.memory': 'Speicherverwaltung',
                'error.not_found': 'Nicht gefunden',
                'error.invalid_input': 'Ungültige Eingabe: {input}',
            },
            'pt': {
                'app.name': 'GhostGPU',
                'app.description': 'Um tempo de execução de computação numérica leve',
                'nav.brand': 'GhostGPU',
                'settings.title': 'Configurações',
                'settings.language': 'Idioma',
                'settings.performance': 'Desempenho',
                'settings.memory': 'Gerenciamento de memória',
                'error.not_found': 'Não encontrado',
                'error.invalid_input': 'Entrada inválida: {input}',
            },
            'ru': {
                'app.name': 'GhostGPU',
                'app.description': 'Легкая среда выполнения численных вычислений',
                'nav.brand': 'GhostGPU',
                'settings.title': 'Параметры',
                'settings.language': 'Язык',
                'settings.performance': 'Производительность',
                'settings.memory': 'Управление памятью',
                'error.not_found': 'Не найдено',
                'error.invalid_input': 'Неверный ввод: {input}',
            },
        }

        # Create locale directory
        self.locale_dir.mkdir(parents=True, exist_ok=True)

        # Write translation files
        for lang_code, translations in languages.items():
            file_path = self.locale_dir / f"{lang_code}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            self.translations[lang_code] = translations

    def set_language(self, language_code: str) -> bool:
        """
        Set current language for translations.

        Parameters
        ----------
        language_code : str
            Language code (e.g., 'en', 'ja', 'zh')

        Returns
        -------
        bool
            True if language was set successfully, False otherwise
        """
        if language_code in self.translations:
            self.current_language = language_code
            return True
        else:
            print(f"Warning: Language '{language_code}' not found. Using '{self.default_language}'")
            self.current_language = self.default_language
            return False

    def get_language(self) -> str:
        """
        Get current language code.

        Returns
        -------
        str
            Current language code
        """
        return self.current_language

    def translate(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """
        Translate a key to the current language with optional parameter substitution.

        Parameters
        ----------
        key : str
            Translation key (e.g., 'app.name')
        default : str, optional
            Default value if translation not found
        **kwargs
            Keyword arguments for string formatting

        Returns
        -------
        str
            Translated string or key if not found
        """
        # Try current language
        if self.current_language in self.translations:
            translations = self.translations[self.current_language]
            if key in translations:
                translation = str(translations[key])
                if kwargs:
                    try:
                        return translation.format(**kwargs)
                    except (KeyError, ValueError):
                        return translation
                return translation

        # Fall back to default language
        if self.default_language in self.translations:
            translations = self.translations[self.default_language]
            if key in translations:
                translation = str(translations[key])
                if kwargs:
                    try:
                        return translation.format(**kwargs)
                    except (KeyError, ValueError):
                        return translation
                return translation

        # Return default or key
        return default if default is not None else key

    def translate_with_params(self, key: str, params: Dict[str, Any]) -> str:
        """
        Translate a key with parameter substitution.

        Parameters
        ----------
        key : str
            Translation key
        params : dict
            Dictionary of parameters for substitution

        Returns
        -------
        str
            Translated and substituted string
        """
        translation = self.translate(key)
        try:
            return translation.format(**params)
        except (KeyError, ValueError):
            return translation

    def get_available_languages(self) -> List[str]:
        """
        Get list of available language codes.

        Returns
        -------
        list
            List of language codes
        """
        return sorted(list(self.translations.keys()))

    def get_missing_translations(self, target_language: str, source_language: str = "en") -> List[str]:
        """
        Find keys that are in source language but missing in target language.

        Parameters
        ----------
        target_language : str
            Target language code
        source_language : str
            Source language code (default: 'en')

        Returns
        -------
        list
            List of missing translation keys
        """
        if source_language not in self.translations or target_language not in self.translations:
            return []

        source_keys = set(self.translations[source_language].keys())
        target_keys = set(self.translations[target_language].keys())
        missing = source_keys - target_keys
        return sorted(list(missing))

    def add_translation(self, language_code: str, key: str, value: str) -> None:
        """
        Add or update a translation.

        Parameters
        ----------
        language_code : str
            Language code
        key : str
            Translation key
        value : str
            Translation value
        """
        if language_code not in self.translations:
            self.translations[language_code] = {}
        self.translations[language_code][key] = value

    def save_translations(self, language_code: str) -> None:
        """
        Save translations to file.

        Parameters
        ----------
        language_code : str
            Language code to save
        """
        if language_code not in self.translations:
            print(f"Warning: Language '{language_code}' not found")
            return

        file_path = self.locale_dir / f"{language_code}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.translations[language_code], f, ensure_ascii=False, indent=2)


# Global instance
_i18n_manager: Optional[I18nManager] = None


def get_i18n_manager(locale_dir: str = "locale") -> I18nManager:
    """
    Get or create the global i18n manager instance.

    Parameters
    ----------
    locale_dir : str
        Directory containing translation JSON files

    Returns
    -------
    I18nManager
        Global i18n manager instance
    """
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager(locale_dir)
    return _i18n_manager


def translate(key: str, default: Optional[str] = None) -> str:
    """
    Convenience function to translate a key using the global manager.

    Parameters
    ----------
    key : str
        Translation key
    default : str, optional
        Default value if not found

    Returns
    -------
    str
        Translated string
    """
    return get_i18n_manager().translate(key, default)


def set_language(language_code: str) -> None:
    """
    Convenience function to set language using the global manager.

    Parameters
    ----------
    language_code : str
        Language code (e.g., 'en', 'ja')
    """
    get_i18n_manager().set_language(language_code)


if __name__ == '__main__':
    # Demo
    manager = get_i18n_manager()
    print("Available languages:", manager.get_available_languages())
    print()

    for lang in manager.get_available_languages():
        manager.set_language(lang)
        print(f"{lang}: {manager.translate('app.name')} - {manager.translate('app.description')}")
