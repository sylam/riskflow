########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of RiskFlow.
#
# RiskFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# RiskFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RiskFlow.  If not, see <http://www.gnu.org/licenses/>.
########################################################################

# standard imports
import os
import inspect
import importlib
import logging
import json # For formatting default values
import shutil
from collections import defaultdict
from pathlib import Path
from riskflow import fields


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper Function to Fetch Documentation ---
def fetch_module_documentation(module_name, attribute='documentation', package='riskflow'):
    """
    Fetches documentation attributes from classes within a specified module.
    Also fetches 'field_desc' specifically from riskfactors.
    """
    document_mapping = defaultdict(list)
    try:
        module = importlib.import_module(module_name, package=package)
        for class_name, cls in inspect.getmembers(module, inspect.isclass):
            # Get 'documentation' attribute
            doc = getattr(cls, attribute, None)
            if doc and isinstance(doc, (list, tuple)) and len(doc) == 2:
                section_name, doc_content = doc
                if isinstance(doc_content, list):
                    document_mapping[section_name].append((class_name, doc_content))
                else:
                    logging.warning(f"Documentation attribute '{attribute}' in class '{class_name}' in module '{module_name}' has unexpected format.")

            # Get 'field_desc' attribute specifically if module is riskfactors
            if module_name == '.riskfactors' and hasattr(cls, 'field_desc'):
                field_desc = getattr(cls, 'field_desc', None)
                if field_desc and isinstance(field_desc, (list, tuple)) and len(field_desc) == 2:
                    section_name, desc_content = field_desc
                    if isinstance(desc_content, list):
                        # Store under a specific key for later use in JSON docs
                        document_mapping['_riskfactor_field_desc'].append((class_name, (section_name, desc_content)))
                    else:
                        logging.warning(f"Attribute 'field_desc' in class '{class_name}' in module '{module_name}' has unexpected format.")

    except ImportError:
        logging.error(f"Could not import module '{module_name}' from package '{package}'.")
    except Exception as e:
        logging.error(f"Error fetching documentation from '{module_name}': {e}")
    return dict(document_mapping)

# --- Main Documentation Generator Class ---
class ConstructMarkdown(object):
    """
    Constructs the markdown (.md) files and mkdocs config by combining
    static content from docs_src/ and dynamic content extracted from the codebase.
    Leverages fields.py for detailed JSON configuration documentation.
    """
    SECTIONS = {
        'Theory': ('Theory', '.stochasticprocess', 'documentation'),
        'Valuation': ('Valuation', '.instruments', 'documentation'),
        'Bootstrapping': ('Bootstrapping', '.bootstrappers', 'documentation'),
        'API_Usage': ('API & Usage', '.calculation', 'documentation'),
        'JSON_Config': ('JSON Configuration', '.riskfactors', 'field_desc') # Special handling now uses fields.py
    }

    def __init__(self, project_dir):
        self.project_dir = Path(project_dir).resolve()
        self.docs_src_dir = self.project_dir / 'docs_src'
        self.doc_build_dir = self.project_dir / 'docs'
        self.package_name = 'riskflow'
        self.static_dirs = ['quickstart', 'running_calcs']
        # Sub‑folders inside docs_src that must be copied wholesale
        if not self.docs_src_dir.is_dir():
            logging.error(f"'docs_src' directory not found at {self.docs_src_dir}")
            raise FileNotFoundError(f"'docs_src' directory not found at {self.docs_src_dir}")
        if not fields:
            logging.error("fields.py mapping is not available. JSON documentation generation will be limited.")


    def _read_static_content(self, filepath):
        """Reads content from a static markdown file."""
        try:
            with open(filepath, 'rt', encoding='utf-8') as f:
                return f.read().splitlines()
        except FileNotFoundError:
            logging.warning(f"Static file not found: {filepath}")
            return []
        except Exception as e:
            logging.error(f"Error reading static file {filepath}: {e}")
            return []

    def _generate_filename(self, page_title):
        """Generates a safe filename from a page title."""
        return page_title.lower().replace(' ', '_').replace('&', 'and').replace('/','_') + '.md'

    def _generate_nav_entry(self, title, filepath):
        """Formats a navigation entry for mkdocs.yml."""
        relative_path = filepath.relative_to(self.doc_build_dir)
        nav_path = str(relative_path).replace(os.sep, '/')
        return f"- '{title}': '{nav_path}'"

    def _infer_data_type(self, meta):
        """Infers data type string from field metadata."""
        widget = meta.get('widget')
        obj = meta.get('obj')
        if obj == 'Percent' or obj == 'Basis' or widget == 'Float' or widget == 'BoundedFloat': return "Number (Float)"
        if widget == 'Integer': return "Number (Integer)"
        if widget == 'DatePicker': return "String (Date: YYYY-MM-DD)"
        if widget == 'Dropdown': return f"String (Allowed: `{', '.join(meta.get('values',[]))}`)"
        if obj == 'Period': return "String (Period, e.g., '3M', '1Y6M')"
        if obj == 'Tuple': return "String (Dot-separated, e.g., 'USD.EUR')"
        # Link complex types
        if widget == 'Flot' or (widget == 'Table' and obj == 'DateValueList'): return "JSON Array (See `.Curve` type in [General Types](../general_types.md))"
        if widget == 'Three': return "JSON Object/Array (See `.Surface`/`.Space` type in [General Types](../general_types.md))"
        if obj == 'DateList': return "JSON Array (See `.DateList` type in [General Types](../general_types.md))"
        if obj == 'DateEqualList': return "JSON Array (See `.DateEqualList` type in [General Types](../general_types.md))"
        if obj == 'CreditSupportList': return "JSON Array (See `.CreditSupportList` type in [General Types](../general_types.md))"
        if widget == 'Table': return "JSON Array of Arrays (See description for column structure)"
        if widget == 'Container': return "JSON Object (Nested Structure)"
        return "String" # Default

    def _format_default_value(self, value, meta):
        """Formats default value nicely."""
        obj = meta.get('obj')
        widget = meta.get('widget')
        try:
            if widget in ['Flot', 'Three', 'Table'] and isinstance(value, str):
                 # Attempt to parse JSON string defaults for complex types for pretty printing
                 parsed = json.loads(value)
                 return json.dumps(parsed, indent=2)
            elif widget == 'DatePicker' and not value:
                 return "`null` or empty string"
            elif value is None:
                 return "`null`"
            elif isinstance(value, (dict, list)):
                 return json.dumps(value, indent=2)
            else:
                 return f"`{value}`"
        except: # Fallback for invalid JSON or other errors
            return f"`{value}`"


    def fetch_and_write_section(self, section_key, display_name, module_name, doc_attribute):
        """ Fetches dynamic docs, reads static docs, merges, and writes the output file. """
        logging.info(f"Processing section: {display_name}")
        nav_entries = {}
        section_src_dir = self.docs_src_dir / section_key.lower()
        section_build_dir = self.doc_build_dir / section_key.lower()
        section_build_dir.mkdir(parents=True, exist_ok=True)

        module_docs = fetch_module_documentation(module_name, doc_attribute, package=self.package_name)

        static_files = {}
        if section_src_dir.is_dir():
            for md_file in sorted(section_src_dir.glob('*.md')): # Sort for consistent order
                 page_key = md_file.stem.replace('_', ' ').title()
                 static_files[page_key] = md_file

        page_titles_ordered = list(static_files.keys())
        processed_dynamic_sections = set()

        for page_title in page_titles_ordered:
            static_filepath = static_files.get(page_title)
            final_doc = []
            if static_filepath:
                final_doc.extend(self._read_static_content(static_filepath))
                final_doc.extend(['', '---', ''])

            dynamic_content_list = module_docs.get(page_title, [])
            if dynamic_content_list:
                 processed_dynamic_sections.add(page_title)
                 for class_name, md_strings in sorted(dynamic_content_list):
                      final_doc.extend(['', f"## `{class_name}`", ''])
                      final_doc.extend(md_strings)

            build_filename = self._generate_filename(page_title)
            build_filepath = section_build_dir / build_filename
            try:
                with open(build_filepath, 'wt', encoding='utf-8') as f:
                    f.write('\n'.join(final_doc))
                nav_entries[page_title] = build_filepath
                logging.info(f"  Generated: {build_filepath}")
            except Exception as e:
                logging.error(f"  Error writing file {build_filepath}: {e}")

        # Process remaining dynamic sections
        for section_title, content_list in module_docs.items():
             if section_title not in processed_dynamic_sections and not section_title.startswith('_'):
                logging.warning(f"  Dynamic documentation found for section '{section_title}' but no corresponding static file in {section_src_dir}. Creating standalone page.")
                final_doc = [f"# {section_title}", ""]
                for class_name, md_strings in sorted(content_list):
                    final_doc.extend(['', f"## `{class_name}`", ''])
                    final_doc.extend(md_strings)

                build_filename = self._generate_filename(section_title)
                build_filepath = section_build_dir / build_filename
                try:
                    with open(build_filepath, 'wt', encoding='utf-8') as f:
                        f.write('\n'.join(final_doc))
                    nav_entries[section_title] = build_filepath
                    logging.info(f"  Generated standalone: {build_filepath}")
                except Exception as e:
                    logging.error(f"  Error writing standalone file {build_filepath}: {e}")

        return {display_name: nav_entries}


    def generate_json_docs(self):
        """Generates detailed documentation for JSON configuration using fields.py."""
        if not fields: return {} # Skip if fields.py wasn't imported

        logging.info("Processing section: JSON Configuration")
        display_name = 'JSON Configuration'
        json_nav_entries = {}
        section_src_dir = self.docs_src_dir / 'json'
        section_build_dir = self.doc_build_dir / 'json'
        section_build_dir.mkdir(parents=True, exist_ok=True)

        # --- Process static overview files ---
        static_order = ['Index', 'General Types', 'System Parameters',
                        'Model Configuration', 'Price Factors Overview', 'Price Models',
                        'Correlations', 'Market Prices', 'Bootstrapper Configuration',
                        'Trade Data'] # Add index.md for the section landing page
        for page_title in static_order:
            filename_base = page_title.lower().replace(' ', '_')
            static_filepath = section_src_dir / (filename_base + '.md')
            build_filepath = section_build_dir / (filename_base + '.md')
            content = self._read_static_content(static_filepath)
            if content:
                 try:
                    with open(build_filepath, 'wt', encoding='utf-8') as f:
                        f.write('\n'.join(content))
                    # Add to nav, handle 'Index' specifically
                    nav_title = 'Overview' if page_title == 'Index' else page_title
                    json_nav_entries[nav_title] = build_filepath
                    logging.info(f"  Generated JSON static: {build_filepath}")
                 except Exception as e:
                    logging.error(f"  Error writing JSON static file {build_filepath}: {e}")

        # --- Generate dynamic pages based on fields.py ---
        sections_to_process = {
            'Price Factors': ('Factor', 'price_factors'),
            'Price Models': ('Process', 'price_models'),
            'Trade Data': ('Instrument', 'trade_data')
            # Add MarketPrices if needed, structure is more complex
        }

        for section_title, (mapping_key, build_subdir) in sections_to_process.items():
            logging.info(f"    Generating JSON details for: {section_title}")
            if mapping_key not in fields.mapping:
                logging.warning(f"      Mapping key '{mapping_key}' not found in fields.py")
                continue

            mapping_data = fields.mapping[mapping_key]
            details_build_dir = section_build_dir / build_subdir
            details_build_dir.mkdir(parents=True, exist_ok=True)
            section_nav = {}

            for item_type, field_identifiers in sorted(mapping_data.get('types', {}).items()):
                page_title = item_type # Use the type name as page title
                build_filename = self._generate_filename(item_type)
                build_filepath = details_build_dir / build_filename
                content = [f"# {item_type}", f"JSON configuration for the `{item_type}` type.", ""]

                # Determine the actual field names to document
                actual_field_names = set()
                if mapping_key == 'Instrument': # Instruments list field *groups*
                    for group_name in field_identifiers:
                        actual_field_names.update(mapping_data.get('sections',{}).get(group_name, []))
                else: # Factors and Processes list field *names* directly
                     actual_field_names.update(field_identifiers)

                # Document each field
                for field_name in sorted(list(actual_field_names)):
                    if field_name in mapping_data.get('fields', {}):
                        meta = mapping_data['fields'][field_name]
                        desc = meta.get('description', field_name)
                        dtype = self._infer_data_type(meta)
                        default_val_str = self._format_default_value(meta.get('value'), meta)

                        content.append(f"### `{field_name}`")
                        content.append(f"- **Description:** {desc}")
                        content.append(f"- **JSON Type:** {dtype}")
                        content.append(f"- **Default:** {default_val_str}")
                        # Handle nested containers if info is available
                        if meta.get('widget') == 'Container' and 'sub_fields' in meta:
                            content.append("- **Nested Fields:**")
                            for sub_field_name in meta['sub_fields']:
                                if sub_field_name in mapping_data.get('fields', {}):
                                     sub_meta = mapping_data['fields'][sub_field_name]
                                     sub_desc = sub_meta.get('description', sub_field_name)
                                     content.append(f"  - `{sub_field_name}` ({sub_desc})")
                                else:
                                     content.append(f"  - `{sub_field_name}`")

                        content.append("")
                    else:
                        logging.warning(f"      Field '{field_name}' listed for type '{item_type}' not found in '{mapping_key}' fields mapping.")

                # Write file and add to nav
                try:
                    with open(build_filepath, 'wt', encoding='utf-8') as f:
                        f.write('\n'.join(content))
                    section_nav[page_title] = build_filepath
                    logging.info(f"      Generated detail doc: {build_filepath}")
                except Exception as e:
                    logging.error(f"      Error writing detail doc file {build_filepath}: {e}")

            # Add the generated sub-navigation to the main JSON navigation
            if section_nav:
                json_nav_entries[section_title] = section_nav

        return {display_name: json_nav_entries}


    def build_config(self, generated_docs_nav):
        """Generates the mkdocs.yml content."""
        config_lines = [
            'site_name: RiskFlow',
            'site_description: Quantitative Finance library using PyTorch for pricing and risk simulation.',
            'site_author: Shuaib Osman',
            'repo_url: https://github.com/sylam/riskflow', # Link to repo
            'repo_name: sylam/riskflow',
            '',
            'theme:',
            '  name: material',
            '  features:',
            '    - navigation.tabs',
            '    - navigation.sections',
            '    - toc.integrate',
            '    - navigation.top',
            '    - search.suggest',
            '    - search.highlight',
            # '    - content.tabs.link', # Uncomment if using tabs in markdown
            '    - content.code.annotation',
            '    - content.code.copy',
            '  language: en',
            '  palette:',
            '    - scheme: default',
            '      toggle:',
            '        icon: material/toggle-switch-off-outline',
            '        name: Switch to dark mode',
            '      primary: teal',
            '      accent: purple',
            '    - scheme: slate',
            '      toggle:',
            '        icon: material/toggle-switch',
            '        name: Switch to light mode',
            '      primary: teal',
            '      accent: lime',
            '',
            'extra:',
            '  social:',
            '    - icon: fontawesome/brands/github', # Correct fontawesome icon syntax
            '      link: https://github.com/sylam/riskflow',
            '',
            'nav:',
            f"    - Home: index.md",
            f"    - Requirements: require.md",
            f"    - Quick Start: quickstart.md",
            f"    - Running Calculations: running_calcs.md",
            f"    - Understanding Output: output.md",
            f"    - API Overview: api_overview.md"
        ]

        base_indent = "    "
        sub_indent = "        "
        sub_sub_indent = "            "
        sub_sub_sub_indent = "                " # For JSON Factor/Model/Trade types

        # Order sections based on SECTIONS definition
        nav_order = ['Theory', 'Valuation', 'Bootstrapping', 'API_Usage', 'JSON_Config']
        for section_key in nav_order:
            if section_key in self.SECTIONS:
                 display_name = self.SECTIONS[section_key][0]
                 if display_name in generated_docs_nav:
                     pages = generated_docs_nav[display_name]
                     config_lines.append(f"{base_indent}- {display_name}:")
                     # Sort pages alphabetically within a section, except for JSON structure
                     page_items = sorted(pages.items()) if display_name != 'JSON Configuration' else pages.items()

                     for page_title, filepath_or_dict in page_items:
                         if isinstance(filepath_or_dict, dict):
                             config_lines.append(f"{sub_indent}- {page_title}:")
                             # Sort sub-pages alphabetically
                             for sub_page_title, sub_filepath_or_dict in sorted(filepath_or_dict.items()):
                                 if isinstance(sub_filepath_or_dict, dict): # e.g., JSON -> Price Factors -> FactorType
                                      config_lines.append(f"{sub_sub_indent}- {sub_page_title}:")
                                      for sub_sub_title, sub_sub_filepath in sorted(sub_filepath_or_dict.items()):
                                          config_lines.append(f"{sub_sub_sub_indent}{self._generate_nav_entry(sub_sub_title, sub_sub_filepath)}")
                                 elif isinstance(sub_filepath_or_dict, Path):
                                      config_lines.append(f"{sub_sub_indent}{self._generate_nav_entry(sub_page_title, sub_filepath_or_dict)}")
                         elif isinstance(filepath_or_dict, Path):
                             config_lines.append(f"{sub_indent}{self._generate_nav_entry(page_title, filepath_or_dict)}")

        config_lines.extend([
            '',
            'extra_javascript:',
            '  - https://polyfill.io/v3/polyfill.min.js?features=es6',
            '  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js',
            '',
            'markdown_extensions:',
            '  - pymdownx.arithmatex:',
            '      generic: true',
            '  - pymdownx.highlight:',
            '      anchor_linenums: true',
            '  - pymdownx.inlinehilite',
            '  - pymdownx.snippets',
            '  - admonition',
            '  - pymdownx.details',
            '  - pymdownx.superfences',
            '  - pymdownx.mark',
            '  - attr_list',
            '  - md_in_html',
            '  - tables'
        ])
        return config_lines

    def build(self):
        """Runs the documentation build process."""
        logging.info(f"Starting documentation build...")
        logging.info(f"Project Root: {self.project_dir}")
        logging.info(f"Source Docs : {self.docs_src_dir}")
        logging.info(f"Build Output: {self.doc_build_dir}")

        self.doc_build_dir.mkdir(parents=True, exist_ok=True)
        generated_nav = {}

        # --- Generate Static Pages ---
        logging.info("Processing static pages...")
        static_pages = ['index.md', 'require.md', 'quickstart.md',
                        'running_calcs.md', 'output.md', 'api_overview.md']
        for page in static_pages:
            src_path = self.docs_src_dir / page
            build_path = self.doc_build_dir / page
            content = self._read_static_content(src_path)
            if content:
                 try:
                     with open(build_path, 'wt', encoding='utf-8') as f: f.write('\n'.join(content))
                     logging.info(f"  Generated: {build_path}")
                 except Exception as e: logging.error(f"  Error writing file {build_path}: {e}")
        # --- Copy whole static directories (quickstart/, running_calcs/) ---

        for d in self.static_dirs:
            src_dir = self.docs_src_dir / d
            dest_dir = self.doc_build_dir / d

            if src_dir.is_dir():
                # Remove any previous artefacts for a clean copy
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(src_dir, dest_dir)
                logging.info(f"  Copied directory: {src_dir}  →  {dest_dir}")
            else:
                logging.warning(f"  Static directory not found: {src_dir}")

        # --- Generate Section Docs (excluding JSON, handled separately) ---
        for section_key, (display_name, module, attr) in self.SECTIONS.items():
            if section_key != 'JSON_Config':
                nav_data = self.fetch_and_write_section(section_key, display_name, module, attr)
                generated_nav.update(nav_data)

        # --- Generate Detailed JSON Docs using fields.py ---
        json_nav_data = self.generate_json_docs()
        generated_nav.update(json_nav_data)

        # --- Generate mkdocs.yml ---
        logging.info("Generating mkdocs.yml...")
        mkdocs_config_content = self.build_config(generated_nav)
        mkdocs_yml_path = self.project_dir / 'mkdocs.yml'
        try:
            with open(mkdocs_yml_path, 'wt', encoding='utf-8') as f: f.write('\n'.join(mkdocs_config_content))
            logging.info(f"Generated: {mkdocs_yml_path}")
        except Exception as e: logging.error(f"Error writing {mkdocs_yml_path}: {e}")

        logging.info("Documentation build script finished.")
        print("\nTo view the documentation:")
        print(f"1. Navigate to the project directory: cd \"{self.project_dir}\"") # Added quotes for paths with spaces
        print(f"2. Run: mkdocs serve")
        print("   (Ensure mkdocs, mkdocs-material, pymdown-extensions are installed: pip install mkdocs mkdocs-material pymdown-extensions)")


# --- Main Execution Block ---
if __name__ == '__main__':
    script_dir = Path(__file__).parent.resolve()
    # Assume documentation.py lives directly inside the project root (sibling to riskflow/ and docs_src/)
    # If it lives *inside* riskflow/, change project_root to script_dir.parent
    project_root = script_dir
    logging.info(f"Detected script directory: {script_dir}")
    logging.info(f"Using project root: {project_root}")

    if not (project_root / 'riskflow').is_dir():
         logging.warning(f"Could not find 'riskflow' directory in {project_root}. Project root might be incorrect.")

    md = ConstructMarkdown(project_root)
    md.build()