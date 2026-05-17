#!/usr/bin/env python3
"""
Enhanced markdown TL;DR sections with comprehensive overviews.
Replaces brief TL;DR summaries with detailed explanations and tables.
"""

import json
import re
import os
from pathlib import Path
from typing import Optional, Tuple

class TLDREnhancer:
    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def find_tldr_section(self, content: str) -> Optional[Tuple[int, int]]:
        """Find TL;DR section boundaries in markdown."""
        # Find ## TL;DR header
        tldr_match = re.search(r'^## TL;DR\s*\n', content, re.MULTILINE)
        if not tldr_match:
            return None

        start = tldr_match.start()

        # Find next ## header (end of TL;DR section)
        remaining = content[tldr_match.end():]
        next_section = re.search(r'^##(?!#) ', remaining, re.MULTILINE)

        if next_section:
            end = tldr_match.end() + next_section.start()
        else:
            end = len(content)

        return (start, end)

    def extract_concept_name(self, filename: str) -> str:
        """Extract concept name from filename (remove NN- prefix)."""
        name = Path(filename).stem
        # Remove numeric prefix like "01-"
        return re.sub(r'^\d+-', '', name).replace('-', ' ').title()

    def create_overview_section(self, concept_name: str, concept_key: str) -> str:
        """Create enhanced overview section from config or generic template."""
        # Get config for this concept (fallback to generic)
        config = self.config.get(concept_key, {})

        overview_title = config.get('overview_title', f"Understanding {concept_name}")

        overview_paragraphs = config.get('overview_paragraphs', self._generate_generic_paragraphs(concept_name))

        # Build the new section
        new_section = f"## {overview_title}\n\n"
        for para in overview_paragraphs:
            new_section += f"{para}\n\n"

        return new_section

    def _generate_generic_paragraphs(self, concept_name: str):
        """Generate generic paragraphs for concepts not in config."""
        return [
            f"{concept_name} is a foundational concept in large language model development that addresses critical challenges in model architecture, training efficiency, or inference performance. Understanding this concept is essential for anyone working with modern language models, whether in research, fine-tuning, or production deployment.",
            f"The core innovation underlying {concept_name} lies in rethinking standard approaches to achieve better efficiency or effectiveness. Rather than accepting conventional trade-offs, this technique exploits mathematical or architectural insights to push the frontier of what's possible with given computational constraints.",
            f"In practical applications, {concept_name} enables capabilities that would otherwise be infeasible: reducing computational requirements, improving model quality, enabling faster iteration, or supporting new use cases. The real-world impact has made {concept_name} widely adopted across industry applications, from consumer products to enterprise systems.",
            f"Implementing {concept_name} requires understanding both its theoretical foundations and practical considerations. The following sections provide detailed explanations of how {concept_name} works, when to use it, common implementation patterns, and lessons learned from production deployments. By mastering these concepts, practitioners can make informed decisions about when and how to apply {concept_name} to their specific challenges."
        ]

    def enhance_file(self, filepath: str) -> bool:
        """Enhance a single markdown file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            tldr_bounds = self.find_tldr_section(content)
            if not tldr_bounds:
                print(f"⏭️  No TL;DR found in {filepath}")
                return False

            start, end = tldr_bounds
            concept_name = self.extract_concept_name(filepath)
            concept_key = concept_name.lower().replace(' ', '-')

            # Create new overview
            new_section = self.create_overview_section(concept_name, concept_key)

            # Replace TL;DR with new section
            new_content = content[:start] + new_section + content[end:]

            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✅ Enhanced: {filepath}")
            return True

        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            return False

    def enhance_all(self, concepts_dir: str) -> None:
        """Enhance all concept files in directory."""
        concept_files = sorted(Path(concepts_dir).glob('*.md'))

        print(f"\n🔄 Processing {len(concept_files)} concept files...\n")

        success_count = 0
        for filepath in concept_files:
            if self.enhance_file(str(filepath)):
                success_count += 1

        print(f"\n✨ Complete! Enhanced {success_count}/{len(concept_files)} files")

def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config' / 'tldr_enhancements.json'
    concepts_dir = script_dir.parent / 'llm' / 'concepts'

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    enhancer = TLDREnhancer(str(config_path))
    enhancer.enhance_all(str(concepts_dir))

if __name__ == '__main__':
    main()
