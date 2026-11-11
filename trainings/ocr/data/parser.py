import csv
import sys
import os
import re
import json
from typing import List, Dict, Any, Optional


def split_into_sections(filepath: str) -> List[List[str]]:
    sections = []
    current_section = []
    collecting_variants = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                
                # Check if this is a "Variants:" line
                if line == 'Variants:':
                    # Start collecting variant lines for the current section
                    collecting_variants = True
                    continue
                
                # If we're collecting variants and line starts with space
                if collecting_variants and line.startswith(' '):
                    # Add variant line to current section
                    current_section.append(line)
                    continue
                
                # If we're collecting variants and encounter a non-indented line
                # (new item section), save current section and start new one
                if collecting_variants and line and not line.startswith(' '):
                    # Save current section if not empty
                    if current_section:
                        sections.append(current_section)
                        current_section = []
                    collecting_variants = False
                
                # Add line to current section
                current_section.append(line)
        
        # Don't forget the last section
        if current_section:
            sections.append(current_section)
            
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return []
    
    return sections


def process_section(section_lines: List[str]) -> Dict[str, Any]:
    item = {
        'item_name': '',
        'item_class': '',
        'properties': {},
        'notes': [],
        'variants': []
    }
    
    # Skip empty lines at the beginning
    lines = [line for line in section_lines if line.strip()]
    if not lines:
        return item
    
    # Find first non-space line (item_name)
    item_name_found = False
    item_class_found = False
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Handle variant lines (start with space)
        if line.startswith(' '):
            item['variants'].append(line.strip())
            i += 1
            continue
        
        # First non-space line is item_name
        if not item_name_found:
            item['item_name'] = line
            item_name_found = True
            i += 1
            continue
        
        # Next non-space line is item_class
        if not item_class_found:
            item['item_class'] = line
            item_class_found = True
            i += 1
            continue
        
        # Process remaining lines
        # Check if this line contains ":"
        if ': ' in line:
            # Split by first ": " to get key and value
            parts = line.split(': ', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                item['properties'][key] = value
        else:
            # Line without ":" goes to notes
            item['notes'].append(line)
        
        i += 1
    
    return item


def parse_items(filepath: str) -> List[Dict[str, Any]]:
    sections = split_into_sections(filepath)
    items = []
    
    print(f"Found {len(sections)} sections")
    
    for i, section in enumerate(sections):
        # Skip empty sections
        if not any(line.strip() for line in section):
            continue
            
        item = process_section(section)
        
        # Only add items that have a name
        if item['item_name']:
            items.append(item)
        else:
            print(f"Warning: Section {i+1} has no item name")
    
    return items


def clean_property_key(key: str) -> str:
    # Remove any non-alphanumeric characters (except underscores)
    cleaned = re.sub(r'[^\w\s]', '', key)
    # Replace spaces with underscores
    cleaned = cleaned.replace(' ', '_')
    # Convert to lowercase
    cleaned = cleaned.lower()
    return cleaned


def write_to_csv(items: List[Dict[str, Any]], output_path: str) -> bool:
    if not items:
        print("No items to write to CSV.")
        return False
    
    # Collect all property keys from all items
    all_property_keys = set()
    for item in items:
        # Clean property keys
        cleaned_props = {}
        for key, value in item['properties'].items():
            cleaned_key = clean_property_key(key)
            cleaned_props[cleaned_key] = value
        item['properties'] = cleaned_props
        all_property_keys.update(cleaned_props.keys())
    
    # Sort property keys alphabetically
    property_keys = sorted(all_property_keys)
    
    # Define CSV columns
    fieldnames = ['item_name', 'item_class'] + property_keys + ['notes', 'variants']
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in items:
                row = {
                    'item_name': item['item_name'],
                    'item_class': item['item_class'],
                    'notes': '; '.join(item['notes']),
                    'variants': '; '.join(item['variants'])
                }
                
                # Add properties
                for key in property_keys:
                    row[key] = item['properties'].get(key, '')
                
                writer.writerow(row)
        
        print(f"Successfully wrote {len(items)} items to '{output_path}'")
        return True
        
    except Exception as e:
        print(f"Error writing CSV file '{output_path}': {e}")
        return False


def write_to_json(items: List[Dict[str, Any]], output_path: str, indent: int = 2) -> bool:
    if not items:
        print("No items to write to JSON.")
        return False
    
    try:
        # Prepare items for JSON output
        json_items = []
        for item in items:
            # Create a clean copy for JSON output
            json_item = {
                'item_name': item['item_name'],
                'item_class': item['item_class'],
                'properties': item['properties'],
                'notes': item['notes'],
                'variants': item['variants']
            }
            json_items.append(json_item)
        
        # Write to JSON file
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_items, jsonfile, indent=indent, ensure_ascii=False)
        
        print(f"Successfully wrote {len(items)} items to '{output_path}'")
        return True
        
    except Exception as e:
        print(f"Error writing JSON file '{output_path}': {e}")
        return False


def main():
    """Main function to run the parser from command line."""
    if len(sys.argv) < 2:
        print("Usage: python new_parser.py <input_file> [output_file]")
        print("Example: python new_parser.py all.txt output.csv")
        print("Example: python new_parser.py all.txt output.json")
        print("If output_file is not specified, uses 'parsed_output.csv'")
        print("Output format is determined by file extension (.csv or .json)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Use default output filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_parsed.csv"
    
    print(f"Parsing '{input_file}'...")
    items = parse_items(input_file)
    
    if items:
        print(f"Parsed {len(items)} items.")
        
        # Determine output format based on file extension
        output_ext = os.path.splitext(output_file)[1].lower()
        
        if output_ext == '.json':
            success = write_to_json(items, output_file)
        else:
            # Default to CSV for any other extension
            success = write_to_csv(items, output_file)
        
        if success:
            print(f"  Total items: {len(items)}")
            names = list(map(lambda i:i['item_name'],items))
            print(names)
            
        print(f"Failed to write output to '{output_file}'")
            
    else:
        print("No items were parsed.")


if __name__ == '__main__':
    main()