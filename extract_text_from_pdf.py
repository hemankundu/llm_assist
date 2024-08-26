import os
import json
import shutil
import git
from PyPDF2 import PdfReader

from config import config_dict

def clone_repositories(repo_list, clone_dir):
    """Clone a list of Git repositories into a specified directory and remove .git directories."""
    cloned_dirs = []
    
    for repo_url in repo_list:
        try:
            repo_name = os.path.splitext(os.path.basename(repo_url))[0]
            repo_path = os.path.join(clone_dir, repo_name)
            git.Repo.clone_from(repo_url, repo_path)
            print(f"Cloned {repo_url} to {repo_path}")

            # Remove the .git directory
            git_dir = os.path.join(repo_path, '.git')
            if os.path.exists(git_dir):
                shutil.rmtree(git_dir)
                print(f"Removed .git directory from {repo_path}")

            cloned_dirs.append(repo_path)

        except Exception as e:
            print(f"Error cloning {repo_url}: {e}")
    
    return cloned_dirs

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text, len(reader.pages)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return "", 0

def save_text_to_file(text, output_path):
    """Save extracted text to a text file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
    except Exception as e:
        print(f"Error saving text to {output_path}: {e}, trying utf-8 only encoding")
        try:
            sanitized_text = text.encode('utf-8', 'ignore').decode('utf-8')
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(sanitized_text)
        except Exception as e:
            print(f"Error saving text to {output_path}: {e}")

def traverse_and_extract(base_dir, output_dir, filename_prefix="text"):
    """Traverse directories, extract text from PDFs, save text files, and maintain metadata."""

    metadata_json_path = os.path.join(output_dir, "metadata.json")
    metadata = []
    text_file_count = 0

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                text, num_pages = extract_text_from_pdf(pdf_path)
                
                # Standardize the text file name
                text_filename = f"{filename_prefix}_{text_file_count}.txt"
                output_path = os.path.join(output_dir, text_filename)
                text_file_count += 1

                # Save the extracted text to a text file
                save_text_to_file(text, output_path)
                
                # Collect metadata
                file_metadata = {
                    "original_pdf_path": pdf_path,
                    "text_file_path": output_path,
                    "num_pages": num_pages
                }
                metadata.append(file_metadata)
                
                print(f"Processed {pdf_path}, saved to {output_path}")

                # Save the metadata to a JSON file
                with open(metadata_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(metadata, json_file, indent=4)

    print(f"Metadata saved to {metadata_json_path}")

def main():

    os.makedirs(config_dict['extract_text_from_pdf']['raw_input_directory'], exist_ok=True)
    os.makedirs(config_dict['extract_text_from_pdf']['text_directory'], exist_ok=True)

    # List of Git repository URLs to clone
    git_repositories = []
    with open(config_dict['extract_text_from_pdf']['git_repo_collections_file']) as  f:
        git_repositories =  f.readlines()
        git_repositories = [x.strip() for x in git_repositories]
        print(f"Found {len(git_repositories)} repos")

    # Clone the repositories
    cloned_dirs = clone_repositories(git_repositories, config_dict['extract_text_from_pdf']['raw_input_directory'])
    print(f"Cloned {len(cloned_dirs)} repos")
    
    # Traverse the directories and extract text from PDFs
    traverse_and_extract(config_dict['extract_text_from_pdf']['raw_input_directory'], config_dict['extract_text_from_pdf']['text_directory'])


if __name__ == "__main__":
    main()