import os
import fitz  # PyMuPDF for PDF processing
from PIL import Image
from pathlib import Path
import shutil

class DocumentProcessor:
    def __init__(self, certificate_folder="./certificates", output_folder="./processed_certificates"):
        self.certificate_folder = certificate_folder
        self.output_folder = output_folder
        self.supported_image_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'}
        self.supported_pdf_formats = {'.pdf'}
        
        # Create folders if they don't exist
        os.makedirs(self.certificate_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
    
    def get_all_files(self):
        """Get all valid files from the certificate folder"""
        if not os.path.exists(self.certificate_folder):
            raise FileNotFoundError(f"Certificate folder '{self.certificate_folder}' not found")
        
        files = []
        for file in os.listdir(self.certificate_folder):
            # Skip system files like .DS_Store, .gitkeep, etc.
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(self.certificate_folder, file)
            if os.path.isfile(file_path):
                files.append(file_path)
        
        if not files:
            raise FileNotFoundError("No valid files found in certificate folder")
        
        return files
    
    def convert_pdf_to_png(self, pdf_path, output_path):
        """Convert PDF to PNG using PyMuPDF"""
        try:
            pdf_document = fitz.open(pdf_path)
            if len(pdf_document) == 0:
                raise Exception("PDF has no pages")
            
            page = pdf_document[0]  # Get first page
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            pix.save(output_path)
            pdf_document.close()
            
            return output_path
        except Exception as e:
            print(f"Error converting PDF to PNG: {e}")
            return None
    
    def convert_image_to_png(self, image_path, output_path):
        """Convert any image format to PNG"""
        try:
            with Image.open(image_path) as img:
                if img.mode in ('RGBA', 'LA'):
                    img.save(output_path, 'PNG')
                elif img.mode == 'P':
                    img = img.convert('RGBA')
                    img.save(output_path, 'PNG')
                else:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_path, 'PNG')
            
            return output_path
        except Exception as e:
            print(f"Error converting image to PNG: {e}")
            return None
    
    def process_single_file(self, file_path):
        """Process a single file and convert to PNG"""
        try:
            file_name = os.path.basename(file_path)
            file_ext = Path(file_path).suffix.lower()
            
            print(f"Processing: {file_name}")
            
            # Generate output filename
            output_filename = f"{Path(file_name).stem}.png"
            output_path = os.path.join(self.output_folder, output_filename)
            
            # Process based on file type
            if file_ext in self.supported_image_formats:
                if file_ext == '.png':
                    shutil.copy2(file_path, output_path)
                    result = output_path
                else:
                    result = self.convert_image_to_png(file_path, output_path)
                    
            elif file_ext in self.supported_pdf_formats:
                result = self.convert_pdf_to_png(file_path, output_path)
            else:
                print(f"Skipping unsupported file format: {file_name}")
                return None
            
            if result and os.path.exists(result) and os.path.getsize(result) > 0:
                print(f"Successfully processed: {output_filename}")
                return result
            else:
                return None
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            return None
    
    def process_documents(self):
        """Main function to process all documents and convert to PNG"""
        try:
            all_files = self.get_all_files()
            processed_files = []
            
            print(f"Found {len(all_files)} file(s) to process")
            
            for file_path in all_files:
                result = self.process_single_file(file_path)
                if result:
                    processed_files.append(result)
            
            if not processed_files:
                print("No files were successfully processed")
                return None
            
            print(f"Successfully processed {len(processed_files)} file(s)")
            
            # Return the first processed file (for backward compatibility)
            return processed_files[0]
                
        except Exception as e:
            print(f"Error processing documents: {e}")
            return None