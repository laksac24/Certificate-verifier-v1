# from typing_extensions import Annotated
# from typing import TypedDict
# from langgraph.graph.message import add_messages
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langgraph.graph import StateGraph, START, END
# import base64
# import os
# import json
# from dotenv import load_dotenv
# from document_processor.doc_processor import DocumentProcessor
# from similar_certificates.similarity import similarity_checker
# from certificate_detection.detector import CertificateDetector  
# from ocr_checking.ocr import ocr_checker
# from database import fetch_data
# import copy
# from PIL import Image
# import io

# load_dotenv()

# certificate_detector = CertificateDetector()

# class State(TypedDict):
#     messages : Annotated[list, add_messages]
#     human : list
#     ecerti : list
#     rejected_certi : list
#     accepted_certi : list
#     ocr_texts : dict

# llm = ChatGroq(model="openai/gpt-oss-120b")
# image_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")

# def resize_image_for_api(image_path, max_size=(1024, 1024), quality=85):
#     """
#     Resize and compress image to reduce file size for API calls
#     """
#     with Image.open(image_path) as img:
#         # Convert to RGB if necessary (for PNG with transparency)
#         if img.mode in ('RGBA', 'LA', 'P'):
#             background = Image.new('RGB', img.size, (255, 255, 255))
#             if img.mode == 'P':
#                 img = img.convert('RGBA')
#             background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
#             img = background
        
#         # Resize image while maintaining aspect ratio
#         img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
#         # Save to bytes buffer with compression
#         buffer = io.BytesIO()
#         img.save(buffer, format='JPEG', quality=quality, optimize=True)
#         buffer.seek(0)
        
#         return buffer.getvalue()


# graph_builder = StateGraph(State)


# def certificate_type_llm(state: State):
#     """
#     Step 1: Process documents and convert all images to PNG
#     Step 2: Classify all PNG files 
#     Step 3: Perform object detection on human-clicked images
#     Step 4: Replace original files with cropped certificates
#     """
    
#     # Step 1: Process documents normally (converts PDFs and images to PNG)
#     print("Step 1: Processing documents and converting to PNG...")
#     processor = DocumentProcessor()
#     processed_image_path = processor.process_documents()
    
#     processed_folder = "./processed_certificates"
#     if not os.path.exists(processed_folder):
#         error_msg = "Processed certificates folder not found."
#         state["messages"].append({"role": "assistant", "content": error_msg})
#         return state
    
#     png_files = [f for f in os.listdir(processed_folder) if f.lower().endswith('.png')]
    
#     if not png_files:
#         error_msg = "No processed PNG files found."
#         state["messages"].append({"role": "assistant", "content": error_msg})
#         return state
    
#     print(f"Found {len(png_files)} processed certificate(s) to classify")
    
#     # Step 2: Classify all PNG files first
#     print("Step 2: Classifying all certificates...")
#     classified_human = []
#     classified_ecerti = []
    
#     for png_file in png_files:
#         image_path = os.path.join(processed_folder, png_file)
        
#         print(f"Classifying: {png_file}")
        
#         try:
#             compressed_image_data = resize_image_for_api(image_path)
#             img_b64 = base64.b64encode(compressed_image_data).decode("utf-8")
            
#             if len(img_b64) > 4_000_000:
#                 print(f"Warning: {png_file} is still large after compression")
#                 compressed_image_data = resize_image_for_api(image_path, max_size=(512, 512), quality=60)
#                 img_b64 = base64.b64encode(compressed_image_data).decode("utf-8")
            
#             # Use LLM classification for all images
#             prompt = [
#                 {"role": "system", "content": "You are an assistant that classifies certificate type. Don't give me any extra information, just tell me whether the certificate is ecertificate or normal human clicked image of the certificate"},
#                 {"role": "user", "content": [
#                     {"type": "text", "text": "Classify this certificate:"},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
#                 ]}
#             ]

#             response = image_llm.invoke(prompt)
#             classification = response.content
            
#             if("ecertificate" in classification.lower()):
#                 classified_ecerti.append(png_file)
#                 print(f"Certificate {png_file} classified as e-certificate")
#             else:
#                 classified_human.append(png_file)
#                 print(f"Certificate {png_file} classified as human-clicked")
            
#             state["messages"].append({
#                 "role": "assistant", 
#                 "content": f"Certificate: {png_file} | Classification: {classification}"
#             })
            
#         except Exception as e:
#             error_msg = f"Error processing {png_file}: {str(e)}"
#             print(error_msg)
#             state["messages"].append({
#                 "role": "assistant", 
#                 "content": error_msg
#             })
#             continue
    
#     # Step 3: Perform object detection on human-clicked images
#     print("Step 3: Performing object detection on human-clicked images...")
#     final_human_certificates = []
    
#     for human_cert in classified_human:
#         image_path = os.path.join(processed_folder, human_cert)
        
#         print(f"Processing {human_cert} for object detection...")
        
#         # Use object detection to crop certificates
#         cropped_paths = certificate_detector.detect_and_crop_certificates(
#             image_path=image_path,
#             output_folder=processed_folder
#         )
        
#         if cropped_paths:
#             print(f"Object detection successful for {human_cert}, found {len(cropped_paths)} certificates")
#             # Remove the original file since we have cropped versions
#             original_path = os.path.join(processed_folder, human_cert)
#             if os.path.exists(original_path):
#                 os.remove(original_path)
#                 print(f"Removed original file: {human_cert}")
            
#             # Add cropped certificates to final list
#             final_human_certificates.extend(cropped_paths)
#         else:
#             print(f"No certificates detected in {human_cert}, keeping original")
#             # If no detection, keep the original file
#             final_human_certificates.append(human_cert)
    
#     # Step 4: Update state with final classifications
#     state["human"] = final_human_certificates
#     state["ecerti"] = classified_ecerti
    
#     print(f"Final classification - Human: {len(final_human_certificates)}, E-certificates: {len(classified_ecerti)}")
#     print(f"Human certificates: {final_human_certificates}")
#     print(f"E-certificates: {classified_ecerti}")
    
#     return state




# def similarity_checking_llm(state:State) :
#     similarity_checker(state)
#     return state

# def ocr_llm(state:State) :
#     ocr_checker(state)
#     return state

# def validation_llm(state:State) :
#     for certi, ocr_text in state["ocr_texts"].items():
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are an assistant that extracts structured fields from OCR text of certificates.
#             Return only valid JSON with fields: EnrollmentNo, Name, Course, CGPA"""),
#             ("user", f"OCR Text: {ocr_text}")
#         ])
        
#         chain = prompt | llm
#         response = chain.invoke({})
#         try:
#             ocr_data = json.loads(response.content)  
#         except:
#             ocr_data = {"EnrollmentNo": None, "Name": None, "Course": None, "CGPA": None}

#         enrollmentNo = ocr_data.get("EnrollmentNo")
#         db_record = fetch_data(str(enrollmentNo)) if enrollmentNo else None
#         if db_record:
#             db_record_copy = copy.deepcopy(db_record)
#             if "_id" in db_record_copy:
#                 db_record_copy["_id"] = str(db_record_copy["_id"])
#             prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", """
#                      you are a checking ai, check whether the data in the given 
#                      in certificate and the data in database are same or not. 
#                      If same then return true, else return false. 
#                      Have a strict checking for key sections like enrollment number and all,
#                      but you can keep a little linient checking for name and other non important sections, but still they muct be very much similar"""),
#                      ("user", "database data:  {db_record_copy}, certificate data : {ocr_data}")
#                 ]
#             )
#             chain = prompt|llm
#             output = chain.invoke({
#                 "db_record_copy" : json.dumps(db_record_copy),
#                 "ocr_data" : json.dumps(ocr_data)
#             })
#             text = output.content.lower()
            
#             # match = (
#             #     str(db_record.get("enrollmentNo")) == str(ocr_data.get("EnrollmentNo")) and
#             #     str(db_record.get("name")).lower() == str(ocr_data.get("Name")).lower() and
#             #     str(db_record.get("course")).lower() == str(ocr_data.get("Course")).lower() and
#             #     float(db_record.get("cgpa")) == float(ocr_data.get("CGPA"))
#             # )
#             if "true" in text:
#                 if certi not in state["accepted_certi"]:
#                     state["accepted_certi"].append(certi)
#                 if certi in state["rejected_certi"]:
#                     state["rejected_certi"].remove(certi)
#             else:
#                 if certi in state["accepted_certi"]:
#                     state["accepted_certi"].remove(certi)
#                 if certi not in state["rejected_certi"]:
#                     state["rejected_certi"].append(certi)
#         else:
#             if certi in state["accepted_certi"]:
#                 state["accepted_certi"].remove(certi)
#             if certi not in state["rejected_certi"]:
#                 state["rejected_certi"].append(certi)
        
#     return state

# def selector_llm(state: State):
#     curr_path = "./processed_certificates"
#     accepted_path = "./accepted_certificates"
#     rejected_path = "./rejected_certificates"

#     os.makedirs(accepted_path, exist_ok=True)
#     os.makedirs(rejected_path, exist_ok=True)

#     for certi in state["accepted_certi"]:
#         src = os.path.join(curr_path, certi)
#         dst = os.path.join(accepted_path, certi)
#         if os.path.exists(src):
#             with Image.open(src) as img:
#                 img.save(dst, 'PNG')

#     for certi in state["rejected_certi"]:
#         src = os.path.join(curr_path, certi)
#         dst = os.path.join(rejected_path, certi)
#         if os.path.exists(src):
#             with Image.open(src) as img:
#                 img.save(dst, 'PNG')

#     return state

# graph_builder.add_node("certificate_type_node", certificate_type_llm)
# graph_builder.add_node("similarity_checking_node", similarity_checking_llm)
# graph_builder.add_node("ocr_node", ocr_llm)
# graph_builder.add_node("validation_node", validation_llm)
# graph_builder.add_node("selector_node", selector_llm)

# graph_builder.add_edge(START,"certificate_type_node")
# graph_builder.add_edge("certificate_type_node","similarity_checking_node")
# graph_builder.add_edge("similarity_checking_node","ocr_node")
# graph_builder.add_edge("ocr_node","validation_node")
# graph_builder.add_edge("validation_node","selector_node")
# graph_builder.add_edge("selector_node",END)


# graph = graph_builder.compile()

# final_state = graph.invoke({"messages": [], "ecerti": [], "human": [], "rejected_certi":[], "accepted_certi":[], "ocr_texts" : {}})

# print("Accepted Cerificate", final_state["accepted_certi"])
# print("Rejected Certificates", final_state["rejected_certi"])
# print("OCR Texts:", final_state["ocr_texts"])


# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# from typing import List, Optional
# import shutil
# import os
# import tempfile
# from pathlib import Path
# import asyncio
# from typing_extensions import Annotated
# from typing import TypedDict
# from langgraph.graph.message import add_messages
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langgraph.graph import StateGraph, START, END
# import base64
# import json
# from dotenv import load_dotenv
# from document_processor.doc_processor import DocumentProcessor
# from similar_certificates.similarity import similarity_checker
# from certificate_detection.detector import CertificateDetector  
# from ocr_checking.ocr import ocr_checker
# from database import fetch_data
# import copy
# from PIL import Image
# import io

# load_dotenv()

# app = FastAPI(title="Certificate Validation API", version="1.0.0")

# # Initialize components (but don't run the pipeline yet)
# certificate_detector = CertificateDetector()

# class State(TypedDict):
#     messages : Annotated[list, add_messages]
#     human : list
#     ecerti : list
#     rejected_certi : list
#     accepted_certi : list
#     ocr_texts : dict

# llm = ChatGroq(model="openai/gpt-oss-120b")
# image_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")

# def resize_image_for_api(image_path, max_size=(1024, 1024), quality=85):
#     """
#     Resize and compress image to reduce file size for API calls
#     """
#     with Image.open(image_path) as img:
#         # Convert to RGB if necessary (for PNG with transparency)
#         if img.mode in ('RGBA', 'LA', 'P'):
#             background = Image.new('RGB', img.size, (255, 255, 255))
#             if img.mode == 'P':
#                 img = img.convert('RGBA')
#             background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
#             img = background
        
#         # Resize image while maintaining aspect ratio
#         img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
#         # Save to bytes buffer with compression
#         buffer = io.BytesIO()
#         img.save(buffer, format='JPEG', quality=quality, optimize=True)
#         buffer.seek(0)
        
#         return buffer.getvalue()

# def certificate_type_llm(state: State):
#     """
#     Step 1: Process documents and convert all images to PNG
#     Step 2: Classify all PNG files 
#     Step 3: Perform object detection on human-clicked images
#     Step 4: Replace original files with cropped certificates
#     """
    
#     # Step 1: Process documents normally (converts PDFs and images to PNG)
#     print("Step 1: Processing documents and converting to PNG...")
#     processor = DocumentProcessor()
#     processed_image_path = processor.process_documents()
    
#     processed_folder = "./processed_certificates"
#     if not os.path.exists(processed_folder):
#         error_msg = "Processed certificates folder not found."
#         # Store as simple string message instead of dict
#         state["messages"].append(error_msg)
#         return state
    
#     png_files = [f for f in os.listdir(processed_folder) if f.lower().endswith('.png')]
    
#     if not png_files:
#         error_msg = "No processed PNG files found."
#         state["messages"].append(error_msg)
#         return state
    
#     print(f"Found {len(png_files)} processed certificate(s) to classify")
    
#     # Step 2: Classify all PNG files first
#     print("Step 2: Classifying all certificates...")
#     classified_human = []
#     classified_ecerti = []
    
#     for png_file in png_files:
#         image_path = os.path.join(processed_folder, png_file)
        
#         print(f"Classifying: {png_file}")
        
#         try:
#             compressed_image_data = resize_image_for_api(image_path)
#             img_b64 = base64.b64encode(compressed_image_data).decode("utf-8")
            
#             if len(img_b64) > 4_000_000:
#                 print(f"Warning: {png_file} is still large after compression")
#                 compressed_image_data = resize_image_for_api(image_path, max_size=(512, 512), quality=60)
#                 img_b64 = base64.b64encode(compressed_image_data).decode("utf-8")
            
#             # Use LLM classification for all images
#             prompt = [
#                 {"role": "system", "content": "You are an assistant that classifies certificate type. Don't give me any extra information, just tell me whether the certificate is ecertificate or normal human clicked image of the certificate"},
#                 {"role": "user", "content": [
#                     {"type": "text", "text": "Classify this certificate:"},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
#                 ]}
#             ]

#             response = image_llm.invoke(prompt)
#             classification = response.content
            
#             if("ecertificate" in classification.lower()):
#                 classified_ecerti.append(png_file)
#                 print(f"Certificate {png_file} classified as e-certificate")
#             else:
#                 classified_human.append(png_file)
#                 print(f"Certificate {png_file} classified as human-clicked")
            
#             # Store as simple string instead of dict
#             message = f"Certificate: {png_file} | Classification: {classification}"
#             state["messages"].append(message)
            
#         except Exception as e:
#             error_msg = f"Error processing {png_file}: {str(e)}"
#             print(error_msg)
#             state["messages"].append(error_msg)
#             continue
    
#     # Step 3: Perform object detection on human-clicked images
#     print("Step 3: Performing object detection on human-clicked images...")
#     final_human_certificates = []
    
#     for human_cert in classified_human:
#         image_path = os.path.join(processed_folder, human_cert)
        
#         print(f"Processing {human_cert} for object detection...")
        
#         # Use object detection to crop certificates
#         cropped_paths = certificate_detector.detect_and_crop_certificates(
#             image_path=image_path,
#             output_folder=processed_folder
#         )
        
#         if cropped_paths:
#             print(f"Object detection successful for {human_cert}, found {len(cropped_paths)} certificates")
#             # Remove the original file since we have cropped versions
#             original_path = os.path.join(processed_folder, human_cert)
#             if os.path.exists(original_path):
#                 os.remove(original_path)
#                 print(f"Removed original file: {human_cert}")
            
#             # Add cropped certificates to final list
#             final_human_certificates.extend(cropped_paths)
#         else:
#             print(f"No certificates detected in {human_cert}, keeping original")
#             # If no detection, keep the original file
#             final_human_certificates.append(human_cert)
    
#     # Step 4: Update state with final classifications
#     state["human"] = final_human_certificates
#     state["ecerti"] = classified_ecerti
    
#     print(f"Final classification - Human: {len(final_human_certificates)}, E-certificates: {len(classified_ecerti)}")
#     print(f"Human certificates: {final_human_certificates}")
#     print(f"E-certificates: {classified_ecerti}")
    
#     return state

# def similarity_checking_llm(state: State):
#     similarity_checker(state)
#     return state

# def ocr_llm(state: State):
#     ocr_checker(state)
#     return state

# def validation_llm(state: State):
#     for certi, ocr_text in state["ocr_texts"].items():
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are an assistant that extracts structured fields from OCR text of certificates.
#             Return only valid JSON with fields: EnrollmentNo, Name, Course, CGPA"""),
#             ("user", f"OCR Text: {ocr_text}")
#         ])
        
#         chain = prompt | llm
#         response = chain.invoke({})
#         try:
#             ocr_data = json.loads(response.content)  
#         except:
#             ocr_data = {"EnrollmentNo": None, "Name": None, "Course": None, "CGPA": None}

#         enrollmentNo = ocr_data.get("EnrollmentNo")
#         db_record = fetch_data(str(enrollmentNo)) if enrollmentNo else None
#         if db_record:
#             db_record_copy = copy.deepcopy(db_record)
#             if "_id" in db_record_copy:
#                 db_record_copy["_id"] = str(db_record_copy["_id"])
#             prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", """
#                      you are a checking ai, check whether the data in the given 
#                      in certificate and the data in database are same or not. 
#                      If same then return true, else return false. 
#                      Have a strict checking for key sections like enrollment number and all,
#                      but you can keep a little linient checking for name and other non important sections, but still they muct be very much similar"""),
#                      ("user", "database data:  {db_record_copy}, certificate data : {ocr_data}")
#                 ]
#             )
#             chain = prompt|llm
#             output = chain.invoke({
#                 "db_record_copy" : json.dumps(db_record_copy),
#                 "ocr_data" : json.dumps(ocr_data)
#             })
#             text = output.content.lower()
            
#             if "true" in text:
#                 if certi not in state["accepted_certi"]:
#                     state["accepted_certi"].append(certi)
#                 if certi in state["rejected_certi"]:
#                     state["rejected_certi"].remove(certi)
#             else:
#                 if certi in state["accepted_certi"]:
#                     state["accepted_certi"].remove(certi)
#                 if certi not in state["rejected_certi"]:
#                     state["rejected_certi"].append(certi)
#         else:
#             if certi in state["accepted_certi"]:
#                 state["accepted_certi"].remove(certi)
#             if certi not in state["rejected_certi"]:
#                 state["rejected_certi"].append(certi)
        
#     return state

# def selector_llm(state: State):
#     curr_path = "./processed_certificates"
#     accepted_path = "./accepted_certificates"
#     rejected_path = "./rejected_certificates"

#     os.makedirs(accepted_path, exist_ok=True)
#     os.makedirs(rejected_path, exist_ok=True)

#     for certi in state["accepted_certi"]:
#         src = os.path.join(curr_path, certi)
#         dst = os.path.join(accepted_path, certi)
#         if os.path.exists(src):
#             with Image.open(src) as img:
#                 img.save(dst, 'PNG')

#     for certi in state["rejected_certi"]:
#         src = os.path.join(curr_path, certi)
#         dst = os.path.join(rejected_path, certi)
#         if os.path.exists(src):
#             with Image.open(src) as img:
#                 img.save(dst, 'PNG')

#     return state

# # Build the graph (but don't execute it yet)
# graph_builder = StateGraph(State)
# graph_builder.add_node("certificate_type_node", certificate_type_llm)
# graph_builder.add_node("similarity_checking_node", similarity_checking_llm)
# graph_builder.add_node("ocr_node", ocr_llm)
# graph_builder.add_node("validation_node", validation_llm)
# graph_builder.add_node("selector_node", selector_llm)

# graph_builder.add_edge(START,"certificate_type_node")
# graph_builder.add_edge("certificate_type_node","similarity_checking_node")
# graph_builder.add_edge("similarity_checking_node","ocr_node")
# graph_builder.add_edge("ocr_node","validation_node")
# graph_builder.add_edge("validation_node","selector_node")
# graph_builder.add_edge("selector_node",END)

# graph = graph_builder.compile()

# def clear_certificates_folder():
#     """Clear the certificates folder before processing new files"""
#     certificates_folder = "./certificates"
#     if os.path.exists(certificates_folder):
#         for filename in os.listdir(certificates_folder):
#             if not filename.startswith('.'):  # Don't delete hidden files like .gitkeep
#                 file_path = os.path.join(certificates_folder, filename)
#                 try:
#                     if os.path.isfile(file_path):
#                         os.unlink(file_path)
#                     elif os.path.isdir(file_path):
#                         shutil.rmtree(file_path)
#                 except Exception as e:
#                     print(f"Error deleting {file_path}: {e}")

# def cleanup_processed_folders():
#     """Clean up processed folders after processing"""
#     folders_to_clean = ["./processed_certificates"]
#     for folder in folders_to_clean:
#         if os.path.exists(folder):
#             for filename in os.listdir(folder):
#                 if not filename.startswith('.'):
#                     file_path = os.path.join(folder, filename)
#                     try:
#                         if os.path.isfile(file_path):
#                             os.unlink(file_path)
#                     except Exception as e:
#                         print(f"Error deleting processed file {file_path}: {e}")

# @app.get("/")
# async def root():
#     return {"message": "Certificate Validation API", "version": "1.0.0"}

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}

# @app.post("/validate-certificates/")
# async def validate_certificates(files: List[UploadFile] = File(...)):
#     """
#     Upload and validate certificates
    
#     Args:
#         files: List of certificate files (PDF, PNG, JPG, etc.)
    
#     Returns:
#         JSON response with validation results
#     """
#     if not files:
#         raise HTTPException(status_code=400, detail="No files provided")
    
#     try:
#         # Clear certificates folder before processing
#         clear_certificates_folder()
        
#         # Ensure certificates folder exists
#         certificates_folder = "./certificates"
#         os.makedirs(certificates_folder, exist_ok=True)
        
#         # Save uploaded files to certificates folder
#         saved_files = []
#         for file in files:
#             if file.filename:
#                 file_path = os.path.join(certificates_folder, file.filename)
#                 with open(file_path, "wb") as buffer:
#                     content = await file.read()
#                     buffer.write(content)
#                 saved_files.append(file.filename)
#                 print(f"Saved file: {file.filename}")
        
#         if not saved_files:
#             raise HTTPException(status_code=400, detail="No valid files were saved")
        
#         print(f"Processing {len(saved_files)} uploaded files...")
        
#         # Run the certificate validation pipeline
#         initial_state = {
#             "messages": [], 
#             "ecerti": [], 
#             "human": [], 
#             "rejected_certi": [], 
#             "accepted_certi": [], 
#             "ocr_texts": {}
#         }
        
#         final_state = graph.invoke(initial_state)
        
#         # Handle processing messages - they should now be simple strings
#         processing_messages = []
#         for msg in final_state["messages"]:
#             if isinstance(msg, str):
#                 processing_messages.append(msg)
#             elif hasattr(msg, 'content'):
#                 # It's an AIMessage object
#                 processing_messages.append(msg.content)
#             elif isinstance(msg, dict) and "content" in msg:
#                 # It's a dictionary
#                 processing_messages.append(msg["content"])
#             else:
#                 # Fallback
#                 processing_messages.append(str(msg))
        
#         # Prepare response
#         response_data = {
#             "status": "completed",
#             "uploaded_files": saved_files,
#             "total_files_processed": len(saved_files),
#             "results": {
#                 "accepted_certificates": final_state["accepted_certi"],
#                 "rejected_certificates": final_state["rejected_certi"],
#                 "total_accepted": len(final_state["accepted_certi"]),
#                 "total_rejected": len(final_state["rejected_certi"]),
#             },
#             "classification": {
#                 "human_certificates": final_state["human"],
#                 "e_certificates": final_state["ecerti"]
#             },
#             "ocr_results": final_state["ocr_texts"],
#             "processing_messages": processing_messages
#         }
        
#         # Clean up processed files (optional)
#         cleanup_processed_folders()
        
#         return JSONResponse(content=response_data, status_code=200)
        
#     except Exception as e:
#         print(f"Error during certificate validation: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# @app.post("/validate-certificates-keep-files/")
# async def validate_certificates_keep_files(files: List[UploadFile] = File(...)):
#     """
#     Upload and validate certificates (keeps processed files for inspection)
    
#     Args:
#         files: List of certificate files (PDF, PNG, JPG, etc.)
    
#     Returns:
#         JSON response with validation results
#     """
#     if not files:
#         raise HTTPException(status_code=400, detail="No files provided")
    
#     try:
#         # Clear certificates folder before processing
#         clear_certificates_folder()
        
#         # Ensure certificates folder exists
#         certificates_folder = "./certificates"
#         os.makedirs(certificates_folder, exist_ok=True)
        
#         # Save uploaded files to certificates folder
#         saved_files = []
#         for file in files:
#             if file.filename:
#                 file_path = os.path.join(certificates_folder, file.filename)
#                 with open(file_path, "wb") as buffer:
#                     content = await file.read()
#                     buffer.write(content)
#                 saved_files.append(file.filename)
#                 print(f"Saved file: {file.filename}")
        
#         if not saved_files:
#             raise HTTPException(status_code=400, detail="No valid files were saved")
        
#         print(f"Processing {len(saved_files)} uploaded files...")
        
#         # Run the certificate validation pipeline
#         initial_state = {
#             "messages": [], 
#             "ecerti": [], 
#             "human": [], 
#             "rejected_certi": [], 
#             "accepted_certi": [], 
#             "ocr_texts": {}
#         }
        
#         final_state = graph.invoke(initial_state)
        
#         # Handle processing messages - they should now be simple strings
#         processing_messages = []
#         for msg in final_state["messages"]:
#             if isinstance(msg, str):
#                 processing_messages.append(msg)
#             elif hasattr(msg, 'content'):
#                 # It's an AIMessage object
#                 processing_messages.append(msg.content)
#             elif isinstance(msg, dict) and "content" in msg:
#                 # It's a dictionary
#                 processing_messages.append(msg["content"])
#             else:
#                 # Fallback
#                 processing_messages.append(str(msg))
        
#         # Prepare response
#         response_data = {
#             "status": "completed",
#             "uploaded_files": saved_files,
#             "total_files_processed": len(saved_files),
#             "results": {
#                 "accepted_certificates": final_state["accepted_certi"],
#                 "rejected_certificates": final_state["rejected_certi"],
#                 "total_accepted": len(final_state["accepted_certi"]),
#                 "total_rejected": len(final_state["rejected_certi"]),
#             },
#             "classification": {
#                 "human_certificates": final_state["human"],
#                 "e_certificates": final_state["ecerti"]
#             },
#             "ocr_results": final_state["ocr_texts"],
#             "processing_messages": processing_messages,
#             "note": "Processed files kept in ./processed_certificates, ./accepted_certificates, and ./rejected_certificates folders"
#         }
        
#         return JSONResponse(content=response_data, status_code=200)
        
#     except Exception as e:
#         print(f"Error during certificate validation: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# @app.delete("/cleanup/")
# async def cleanup_all():
#     """Clean up all processing folders"""
#     try:
#         folders_to_clean = [
#             "./certificates", 
#             "./processed_certificates", 
#             "./accepted_certificates", 
#             "./rejected_certificates"
#         ]
        
#         cleaned_folders = []
#         for folder in folders_to_clean:
#             if os.path.exists(folder):
#                 for filename in os.listdir(folder):
#                     if not filename.startswith('.'):  # Keep hidden files like .gitkeep
#                         file_path = os.path.join(folder, filename)
#                         try:
#                             if os.path.isfile(file_path):
#                                 os.unlink(file_path)
#                         except Exception as e:
#                             print(f"Error deleting {file_path}: {e}")
#                 cleaned_folders.append(folder)
        
#         return {"status": "success", "cleaned_folders": cleaned_folders}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List, Optional
# import shutil
# import os
# import tempfile
# from pathlib import Path
# import asyncio
# from typing_extensions import Annotated
# from typing import TypedDict
# import base64
# import json
# import sys
# import logging
# from dotenv import load_dotenv
# import copy
# from PIL import Image
# import io

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Validate required environment variables
# required_env_vars = ["MONGO_URI"]
# missing_vars = [var for var in required_env_vars if not os.getenv(var)]
# if missing_vars:
#     logger.error(f"Missing required environment variables: {missing_vars}")
#     sys.exit(1)

# app = FastAPI(
#     title="Certificate Validation API", 
#     version="1.0.0",
#     description="API for validating and processing certificates"
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure this properly for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variables for lazy loading
# certificate_detector = None
# llm = None
# image_llm = None
# graph = None

# def get_base_dir():
#     """Get the base directory for file operations"""
#     return Path.cwd()

# def ensure_directories():
#     """Ensure all required directories exist"""
#     base_dir = get_base_dir()
#     dirs = [
#         "certificates",
#         "processed_certificates", 
#         "accepted_certificates",
#         "rejected_certificates",
#         "similar_certificates"
#     ]
    
#     for dir_name in dirs:
#         dir_path = base_dir / dir_name
#         dir_path.mkdir(exist_ok=True)
#         logger.info(f"Ensured directory exists: {dir_path}")

# def initialize_components():
#     """Initialize all components with proper error handling"""
#     global certificate_detector, llm, image_llm, graph
    
#     try:
#         # Import here to handle missing dependencies gracefully
#         from langgraph.graph.message import add_messages
#         from langchain_groq import ChatGroq
#         from langchain_core.prompts import ChatPromptTemplate
#         from langgraph.graph import StateGraph, START, END
        
#         logger.info("Initializing LLM components...")
        
#         # Initialize LLMs
#         llm = ChatGroq(model="openai/gpt-oss-120b")
#         image_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
        
#         # Initialize certificate detector
#         try:
#             from certificate_detection.detector import CertificateDetector
#             certificate_detector = CertificateDetector()
#             logger.info("Certificate detector initialized successfully")
#         except Exception as e:
#             logger.warning(f"Certificate detector initialization failed: {e}")
#             certificate_detector = None
        
#         # Build the graph
#         class State(TypedDict):
#             messages : Annotated[list, add_messages]
#             human : list
#             ecerti : list
#             rejected_certi : list
#             accepted_certi : list
#             ocr_texts : dict
        
#         graph_builder = StateGraph(State)
#         graph_builder.add_node("certificate_type_node", certificate_type_llm)
#         graph_builder.add_node("similarity_checking_node", similarity_checking_llm)
#         graph_builder.add_node("ocr_node", ocr_llm)
#         graph_builder.add_node("validation_node", validation_llm)
#         graph_builder.add_node("selector_node", selector_llm)
        
#         graph_builder.add_edge(START,"certificate_type_node")
#         graph_builder.add_edge("certificate_type_node","similarity_checking_node")
#         graph_builder.add_edge("similarity_checking_node","ocr_node")
#         graph_builder.add_edge("ocr_node","validation_node")
#         graph_builder.add_edge("validation_node","selector_node")
#         graph_builder.add_edge("selector_node",END)
        
#         graph = graph_builder.compile()
#         logger.info("Graph compiled successfully")
        
#         return True
#     except Exception as e:
#         logger.error(f"Failed to initialize components: {e}")
#         return False

# def resize_image_for_api(image_path, max_size=(1024, 1024), quality=85):
#     """Resize and compress image to reduce file size for API calls"""
#     try:
#         with Image.open(image_path) as img:
#             # Convert to RGB if necessary
#             if img.mode in ('RGBA', 'LA', 'P'):
#                 background = Image.new('RGB', img.size, (255, 255, 255))
#                 if img.mode == 'P':
#                     img = img.convert('RGBA')
#                 background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
#                 img = background
            
#             # Resize image while maintaining aspect ratio
#             img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
#             # Save to bytes buffer with compression
#             buffer = io.BytesIO()
#             img.save(buffer, format='JPEG', quality=quality, optimize=True)
#             buffer.seek(0)
            
#             return buffer.getvalue()
#     except Exception as e:
#         logger.error(f"Error resizing image {image_path}: {e}")
#         raise

# def certificate_type_llm(state):
#     """Process and classify certificates"""
#     try:
#         logger.info("Step 1: Processing documents and converting to PNG...")
        
#         # Import here to handle missing dependencies
#         from document_processor.doc_processor import DocumentProcessor
        
#         processor = DocumentProcessor()
#         processed_image_path = processor.process_documents()
        
#         base_dir = get_base_dir()
#         processed_folder = base_dir / "processed_certificates"
        
#         if not processed_folder.exists():
#             error_msg = "Processed certificates folder not found."
#             state["messages"].append(error_msg)
#             return state
        
#         png_files = [f.name for f in processed_folder.glob("*.png")]
        
#         if not png_files:
#             error_msg = "No processed PNG files found."
#             state["messages"].append(error_msg)
#             return state
        
#         logger.info(f"Found {len(png_files)} processed certificate(s) to classify")
        
#         # Classify certificates
#         classified_human = []
#         classified_ecerti = []
        
#         for png_file in png_files:
#             image_path = processed_folder / png_file
            
#             logger.info(f"Classifying: {png_file}")
            
#             try:
#                 compressed_image_data = resize_image_for_api(str(image_path))
#                 img_b64 = base64.b64encode(compressed_image_data).decode("utf-8")
                
#                 if len(img_b64) > 4_000_000:
#                     logger.warning(f"{png_file} is still large after compression")
#                     compressed_image_data = resize_image_for_api(str(image_path), max_size=(512, 512), quality=60)
#                     img_b64 = base64.b64encode(compressed_image_data).decode("utf-8")
                
#                 # Use LLM classification
#                 prompt = [
#                     {"role": "system", "content": "You are an assistant that classifies certificate type. Don't give me any extra information, just tell me whether the certificate is ecertificate or normal human clicked image of the certificate"},
#                     {"role": "user", "content": [
#                         {"type": "text", "text": "Classify this certificate:"},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
#                     ]}
#                 ]

#                 response = image_llm.invoke(prompt)
#                 classification = response.content
                
#                 if "ecertificate" in classification.lower():
#                     classified_ecerti.append(png_file)
#                     logger.info(f"Certificate {png_file} classified as e-certificate")
#                 else:
#                     classified_human.append(png_file)
#                     logger.info(f"Certificate {png_file} classified as human-clicked")
                
#                 message = f"Certificate: {png_file} | Classification: {classification}"
#                 state["messages"].append(message)
                
#             except Exception as e:
#                 error_msg = f"Error processing {png_file}: {str(e)}"
#                 logger.error(error_msg)
#                 state["messages"].append(error_msg)
#                 continue
        
#         # Perform object detection on human-clicked images
#         logger.info("Step 3: Performing object detection on human-clicked images...")
#         final_human_certificates = []
        
#         if certificate_detector:
#             for human_cert in classified_human:
#                 image_path = processed_folder / human_cert
                
#                 logger.info(f"Processing {human_cert} for object detection...")
                
#                 cropped_paths = certificate_detector.detect_and_crop_certificates(
#                     image_path=str(image_path),
#                     output_folder=str(processed_folder)
#                 )
                
#                 if cropped_paths:
#                     logger.info(f"Object detection successful for {human_cert}, found {len(cropped_paths)} certificates")
#                     # Remove original file
#                     if image_path.exists():
#                         image_path.unlink()
#                         logger.info(f"Removed original file: {human_cert}")
                    
#                     final_human_certificates.extend(cropped_paths)
#                 else:
#                     logger.info(f"No certificates detected in {human_cert}, keeping original")
#                     final_human_certificates.append(human_cert)
#         else:
#             logger.warning("Certificate detector not available, skipping object detection")
#             final_human_certificates = classified_human
        
#         # Update state
#         state["human"] = final_human_certificates
#         state["ecerti"] = classified_ecerti
        
#         logger.info(f"Final classification - Human: {len(final_human_certificates)}, E-certificates: {len(classified_ecerti)}")
        
#         return state
        
#     except Exception as e:
#         error_msg = f"Error in certificate_type_llm: {str(e)}"
#         logger.error(error_msg)
#         state["messages"].append(error_msg)
#         return state

# def similarity_checking_llm(state):
#     """Check similarity of certificates"""
#     try:
#         from similar_certificates.similarity import similarity_checker
#         similarity_checker(state)
#         return state
#     except Exception as e:
#         error_msg = f"Error in similarity checking: {str(e)}"
#         logger.error(error_msg)
#         state["messages"].append(error_msg)
#         return state

# def ocr_llm(state):
#     """Perform OCR on certificates"""
#     try:
#         from ocr_checking.ocr import ocr_checker
#         ocr_checker(state)
#         return state
#     except Exception as e:
#         error_msg = f"Error in OCR processing: {str(e)}"
#         logger.error(error_msg)
#         state["messages"].append(error_msg)
#         return state

# def validation_llm(state):
#     """Validate certificates against database"""
#     try:
#         from database import fetch_data
#         from langchain_core.prompts import ChatPromptTemplate
        
#         for certi, ocr_text in state["ocr_texts"].items():
#             prompt = ChatPromptTemplate.from_messages([
#                 ("system", """You are an assistant that extracts structured fields from OCR text of certificates.
#                 Return only valid JSON with fields: EnrollmentNo, Name, Course, CGPA"""),
#                 ("user", f"OCR Text: {ocr_text}")
#             ])
            
#             chain = prompt | llm
#             response = chain.invoke({})
#             try:
#                 ocr_data = json.loads(response.content)  
#             except:
#                 ocr_data = {"EnrollmentNo": None, "Name": None, "Course": None, "CGPA": None}

#             enrollmentNo = ocr_data.get("EnrollmentNo")
#             db_record = fetch_data(str(enrollmentNo)) if enrollmentNo else None
            
#             if db_record:
#                 db_record_copy = copy.deepcopy(db_record)
#                 if "_id" in db_record_copy:
#                     db_record_copy["_id"] = str(db_record_copy["_id"])
                    
#                 prompt = ChatPromptTemplate.from_messages([
#                     ("system", """You are a checking ai, check whether the data in the given 
#                      certificate and the data in database are same or not. 
#                      If same then return true, else return false. 
#                      Have strict checking for key sections like enrollment number,
#                      but you can be lenient for name and other non-important sections."""),
#                     ("user", "database data: {db_record_copy}, certificate data: {ocr_data}")
#                 ])
                
#                 chain = prompt | llm
#                 output = chain.invoke({
#                     "db_record_copy": json.dumps(db_record_copy),
#                     "ocr_data": json.dumps(ocr_data)
#                 })
#                 text = output.content.lower()
                
#                 if "true" in text:
#                     if certi not in state["accepted_certi"]:
#                         state["accepted_certi"].append(certi)
#                     if certi in state["rejected_certi"]:
#                         state["rejected_certi"].remove(certi)
#                 else:
#                     if certi in state["accepted_certi"]:
#                         state["accepted_certi"].remove(certi)
#                     if certi not in state["rejected_certi"]:
#                         state["rejected_certi"].append(certi)
#             else:
#                 if certi in state["accepted_certi"]:
#                     state["accepted_certi"].remove(certi)
#                 if certi not in state["rejected_certi"]:
#                     state["rejected_certi"].append(certi)
        
#         return state
        
#     except Exception as e:
#         error_msg = f"Error in validation: {str(e)}"
#         logger.error(error_msg)
#         state["messages"].append(error_msg)
#         return state

# def selector_llm(state):
#     """Move certificates to appropriate folders"""
#     try:
#         base_dir = get_base_dir()
#         curr_path = base_dir / "processed_certificates"
#         accepted_path = base_dir / "accepted_certificates"
#         rejected_path = base_dir / "rejected_certificates"

#         accepted_path.mkdir(exist_ok=True)
#         rejected_path.mkdir(exist_ok=True)

#         for certi in state["accepted_certi"]:
#             src = curr_path / certi
#             dst = accepted_path / certi
#             if src.exists():
#                 with Image.open(src) as img:
#                     img.save(dst, 'PNG')

#         for certi in state["rejected_certi"]:
#             src = curr_path / certi
#             dst = rejected_path / certi
#             if src.exists():
#                 with Image.open(src) as img:
#                     img.save(dst, 'PNG')

#         return state
        
#     except Exception as e:
#         error_msg = f"Error in selector: {str(e)}"
#         logger.error(error_msg)
#         state["messages"].append(error_msg)
#         return state

# def clear_certificates_folder():
#     """Clear the certificates folder before processing new files"""
#     try:
#         base_dir = get_base_dir()
#         certificates_folder = base_dir / "certificates"
        
#         if certificates_folder.exists():
#             for file_path in certificates_folder.iterdir():
#                 if not file_path.name.startswith('.'):  # Keep hidden files
#                     try:
#                         if file_path.is_file():
#                             file_path.unlink()
#                         elif file_path.is_dir():
#                             shutil.rmtree(file_path)
#                     except Exception as e:
#                         logger.error(f"Error deleting {file_path}: {e}")
#     except Exception as e:
#         logger.error(f"Error clearing certificates folder: {e}")

# def cleanup_processed_folders():
#     """Clean up processed folders after processing"""
#     try:
#         base_dir = get_base_dir()
#         folders_to_clean = [base_dir / "processed_certificates"]
        
#         for folder in folders_to_clean:
#             if folder.exists():
#                 for file_path in folder.iterdir():
#                     if not file_path.name.startswith('.'):
#                         try:
#                             if file_path.is_file():
#                                 file_path.unlink()
#                         except Exception as e:
#                             logger.error(f"Error deleting processed file {file_path}: {e}")
#     except Exception as e:
#         logger.error(f"Error cleaning up processed folders: {e}")

# @app.on_event("startup")
# async def startup_event():
#     """Initialize components on startup"""
#     logger.info("Starting up Certificate Validation API...")
#     ensure_directories()
    
#     success = initialize_components()
#     if not success:
#         logger.error("Failed to initialize some components - API may have limited functionality")
#     else:
#         logger.info("All components initialized successfully")

# @app.get("/")
# async def root():
#     return {
#         "message": "Certificate Validation API", 
#         "version": "1.0.0",
#         "status": "running"
#     }

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     health_status = {
#         "status": "healthy",
#         "components": {
#             "llm": llm is not None,
#             "image_llm": image_llm is not None,
#             "certificate_detector": certificate_detector is not None,
#             "graph": graph is not None
#         }
#     }
#     return health_status

# @app.post("/validate-certificates/")
# async def validate_certificates(files: List[UploadFile] = File(...)):
#     """Upload and validate certificates"""
#     if not files:
#         raise HTTPException(status_code=400, detail="No files provided")
    
#     if not graph:
#         raise HTTPException(status_code=503, detail="Service not properly initialized")
    
#     try:
#         clear_certificates_folder()
        
#         base_dir = get_base_dir()
#         certificates_folder = base_dir / "certificates"
#         certificates_folder.mkdir(exist_ok=True)
        
#         # Save uploaded files
#         saved_files = []
#         for file in files:
#             if file.filename:
#                 file_path = certificates_folder / file.filename
#                 with open(file_path, "wb") as buffer:
#                     content = await file.read()
#                     buffer.write(content)
#                 saved_files.append(file.filename)
#                 logger.info(f"Saved file: {file.filename}")
        
#         if not saved_files:
#             raise HTTPException(status_code=400, detail="No valid files were saved")
        
#         logger.info(f"Processing {len(saved_files)} uploaded files...")
        
#         # Run the pipeline
#         initial_state = {
#             "messages": [], 
#             "ecerti": [], 
#             "human": [], 
#             "rejected_certi": [], 
#             "accepted_certi": [], 
#             "ocr_texts": {}
#         }
        
#         final_state = graph.invoke(initial_state)
        
#         # Process messages
#         processing_messages = []
#         for msg in final_state["messages"]:
#             if isinstance(msg, str):
#                 processing_messages.append(msg)
#             elif hasattr(msg, 'content'):
#                 processing_messages.append(msg.content)
#             elif isinstance(msg, dict) and "content" in msg:
#                 processing_messages.append(msg["content"])
#             else:
#                 processing_messages.append(str(msg))
        
#         response_data = {
#             "status": "completed",
#             "uploaded_files": saved_files,
#             "total_files_processed": len(saved_files),
#             "results": {
#                 "accepted_certificates": final_state["accepted_certi"],
#                 "rejected_certificates": final_state["rejected_certi"],
#                 "total_accepted": len(final_state["accepted_certi"]),
#                 "total_rejected": len(final_state["rejected_certi"]),
#             },
#             "classification": {
#                 "human_certificates": final_state["human"],
#                 "e_certificates": final_state["ecerti"]
#             },
#             "ocr_results": final_state["ocr_texts"],
#             "processing_messages": processing_messages
#         }
        
#         cleanup_processed_folders()
        
#         return JSONResponse(content=response_data, status_code=200)
        
#     except Exception as e:
#         logger.error(f"Error during certificate validation: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# @app.delete("/cleanup/")
# async def cleanup_all():
#     """Clean up all processing folders"""
#     try:
#         base_dir = get_base_dir()
#         folders_to_clean = [
#             "certificates", 
#             "processed_certificates", 
#             "accepted_certificates", 
#             "rejected_certificates"
#         ]
        
#         cleaned_folders = []
#         for folder_name in folders_to_clean:
#             folder = base_dir / folder_name
#             if folder.exists():
#                 for file_path in folder.iterdir():
#                     if not file_path.name.startswith('.'):
#                         try:
#                             if file_path.is_file():
#                                 file_path.unlink()
#                         except Exception as e:
#                             logger.error(f"Error deleting {file_path}: {e}")
#                 cleaned_folders.append(folder_name)
        
#         return {"status": "success", "cleaned_folders": cleaned_folders}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))



from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
import os
import tempfile
from pathlib import Path
import asyncio
from typing_extensions import Annotated
from typing import TypedDict
import base64
import json
import sys
import logging
from dotenv import load_dotenv
import copy
from PIL import Image
import io
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = ["MONGO_URI"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {missing_vars}")
    sys.exit(1)

app = FastAPI(
    title="Certificate Validation API", 
    version="1.0.0",
    description="API for validating and processing certificates"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for components - loaded lazily
certificate_detector = None
llm = None
image_llm = None
graph = None
ocr_reader = None
components_loaded = False
executor = ThreadPoolExecutor(max_workers=4)

# Processing status storage (use Redis in production)
processing_status = {}

class ProcessingStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

def get_base_dir():
    """Get the base directory for file operations"""
    return Path.cwd()

def ensure_directories():
    """Ensure all required directories exist"""
    base_dir = get_base_dir()
    dirs = [
        "certificates",
        "processed_certificates", 
        "accepted_certificates",
        "rejected_certificates",
        "similar_certificates"
    ]
    
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

def initialize_llm_components():
    """Initialize only LLM components quickly"""
    global llm, image_llm, graph
    
    try:
        logger.info("Initializing LLM components...")
        
        # Import here to handle missing dependencies gracefully
        from langgraph.graph.message import add_messages
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph.graph import StateGraph, START, END
        
        llm = ChatGroq(model="openai/gpt-oss-120b")
        image_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
        
        # Build the graph
        class State(TypedDict):
            messages : Annotated[list, add_messages]
            human : list
            ecerti : list
            rejected_certi : list
            accepted_certi : list
            ocr_texts : dict
            processing_id : str
        
        graph_builder = StateGraph(State)
        graph_builder.add_node("certificate_type_node", certificate_type_llm)
        graph_builder.add_node("similarity_checking_node", similarity_checking_llm)
        graph_builder.add_node("ocr_node", ocr_llm)
        graph_builder.add_node("validation_node", validation_llm)
        graph_builder.add_node("selector_node", selector_llm)
        
        graph_builder.add_edge(START,"certificate_type_node")
        graph_builder.add_edge("certificate_type_node","similarity_checking_node")
        graph_builder.add_edge("similarity_checking_node","ocr_node")
        graph_builder.add_edge("ocr_node","validation_node")
        graph_builder.add_edge("validation_node","selector_node")
        graph_builder.add_edge("selector_node",END)
        
        graph = graph_builder.compile()
        logger.info("LLM components initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM components: {e}")
        return False

def initialize_heavy_components():
    """Initialize heavy components (OCR, Certificate Detector) lazily"""
    global certificate_detector, ocr_reader, components_loaded
    
    try:
        logger.info("Initializing heavy components...")
        
        # Initialize certificate detector
        try:
            from certificate_detection.detector import CertificateDetector
            certificate_detector = CertificateDetector()
            logger.info("Certificate detector initialized successfully")
        except Exception as e:
            logger.warning(f"Certificate detector initialization failed: {e}")
            certificate_detector = None
        
        # Initialize OCR reader
        try:
            import easyocr
            logger.info("Starting OCR reader initialization (this may take a few minutes)...")
            ocr_reader = easyocr.Reader(['en'])
            logger.info("OCR reader initialized successfully")
        except Exception as e:
            logger.warning(f"OCR reader initialization failed: {e}")
            ocr_reader = None
        
        components_loaded = True
        logger.info("All heavy components initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize heavy components: {e}")
        return False

def get_ocr_reader():
    """Get OCR reader, initialize if needed"""
    global ocr_reader
    
    if ocr_reader is None:
        try:
            import easyocr
            logger.info("Initializing OCR reader on demand...")
            ocr_reader = easyocr.Reader(['en'])
            logger.info("OCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR reader: {e}")
            raise Exception("OCR reader could not be initialized")
    
    return ocr_reader

def resize_image_for_api(image_path, max_size=(1024, 1024), quality=85):
    """Resize and compress image to reduce file size for API calls"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Resize image while maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save to bytes buffer with compression
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error resizing image {image_path}: {e}")
        raise

def update_processing_status(processing_id: str, status: str, message: str = "", data: dict = None):
    """Update processing status"""
    processing_status[processing_id] = {
        "status": status,
        "message": message,
        "timestamp": time.time(),
        "data": data or {}
    }

def certificate_type_llm(state):
    """Process and classify certificates with timeout handling"""
    processing_id = state.get("processing_id", "unknown")
    
    try:
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, "Processing documents...")
        
        logger.info("Step 1: Processing documents and converting to PNG...")
        
        # Import here to handle missing dependencies
        from document_processor.doc_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        processed_image_path = processor.process_documents()
        
        base_dir = get_base_dir()
        processed_folder = base_dir / "processed_certificates"
        
        if not processed_folder.exists():
            error_msg = "Processed certificates folder not found."
            state["messages"].append(error_msg)
            return state
        
        png_files = [f.name for f in processed_folder.glob("*.png")]
        
        if not png_files:
            error_msg = "No processed PNG files found."
            state["messages"].append(error_msg)
            return state
        
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, f"Classifying {len(png_files)} certificates...")
        
        logger.info(f"Found {len(png_files)} processed certificate(s) to classify")
        
        # Classify certificates with timeout
        classified_human = []
        classified_ecerti = []
        
        for i, png_file in enumerate(png_files):
            update_processing_status(processing_id, ProcessingStatus.PROCESSING, f"Classifying certificate {i+1}/{len(png_files)}")
            
            image_path = processed_folder / png_file
            
            logger.info(f"Classifying: {png_file}")
            
            try:
                # Add timeout for image processing
                future = executor.submit(classify_single_image, str(image_path))
                classification = future.result(timeout=30)  # 30 second timeout per image
                
                if "ecertificate" in classification.lower():
                    classified_ecerti.append(png_file)
                    logger.info(f"Certificate {png_file} classified as e-certificate")
                else:
                    classified_human.append(png_file)
                    logger.info(f"Certificate {png_file} classified as human-clicked")
                
                message = f"Certificate: {png_file} | Classification: {classification}"
                state["messages"].append(message)
                
            except Exception as e:
                error_msg = f"Error processing {png_file}: {str(e)}"
                logger.error(error_msg)
                state["messages"].append(error_msg)
                continue
        
        # Perform object detection with timeout
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, "Performing object detection...")
        logger.info("Step 3: Performing object detection on human-clicked images...")
        final_human_certificates = []
        
        if certificate_detector:
            for human_cert in classified_human:
                try:
                    future = executor.submit(process_object_detection, str(processed_folder / human_cert), str(processed_folder))
                    cropped_paths = future.result(timeout=60)  # 1 minute timeout for object detection
                    
                    if cropped_paths:
                        logger.info(f"Object detection successful for {human_cert}, found {len(cropped_paths)} certificates")
                        # Remove original file
                        original_path = processed_folder / human_cert
                        if original_path.exists():
                            original_path.unlink()
                            logger.info(f"Removed original file: {human_cert}")
                        
                        final_human_certificates.extend(cropped_paths)
                    else:
                        logger.info(f"No certificates detected in {human_cert}, keeping original")
                        final_human_certificates.append(human_cert)
                        
                except Exception as e:
                    logger.error(f"Object detection failed for {human_cert}: {e}")
                    final_human_certificates.append(human_cert)
        else:
            logger.warning("Certificate detector not available, skipping object detection")
            final_human_certificates = classified_human
        
        # Update state
        state["human"] = final_human_certificates
        state["ecerti"] = classified_ecerti
        
        logger.info(f"Final classification - Human: {len(final_human_certificates)}, E-certificates: {len(classified_ecerti)}")
        
        return state
        
    except Exception as e:
        error_msg = f"Error in certificate_type_llm: {str(e)}"
        logger.error(error_msg)
        state["messages"].append(error_msg)
        update_processing_status(processing_id, ProcessingStatus.FAILED, error_msg)
        return state

def classify_single_image(image_path):
    """Classify a single image - runs in thread pool"""
    compressed_image_data = resize_image_for_api(image_path)
    img_b64 = base64.b64encode(compressed_image_data).decode("utf-8")
    
    if len(img_b64) > 4_000_000:
        compressed_image_data = resize_image_for_api(image_path, max_size=(512, 512), quality=60)
        img_b64 = base64.b64encode(compressed_image_data).decode("utf-8")
    
    prompt = [
        {"role": "system", "content": "You are an assistant that classifies certificate type. Don't give me any extra information, just tell me whether the certificate is ecertificate or normal human clicked image of the certificate"},
        {"role": "user", "content": [
            {"type": "text", "text": "Classify this certificate:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}
    ]
    
    response = image_llm.invoke(prompt)
    return response.content

def process_object_detection(image_path, output_folder):
    """Process object detection for a single image - runs in thread pool"""
    if certificate_detector:
        return certificate_detector.detect_and_crop_certificates(
            image_path=image_path,
            output_folder=output_folder
        )
    return []

def similarity_checking_llm(state):
    """Check similarity of certificates with timeout"""
    processing_id = state.get("processing_id", "unknown")
    
    try:
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, "Checking certificate similarity...")
        
        # Add timeout for similarity checking
        future = executor.submit(run_similarity_check, state)
        result_state = future.result(timeout=120)  # 2 minute timeout
        
        return result_state
        
    except Exception as e:
        error_msg = f"Error in similarity checking: {str(e)}"
        logger.error(error_msg)
        state["messages"].append(error_msg)
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, "Similarity check failed, continuing...")
        return state

def run_similarity_check(state):
    """Run similarity check - runs in thread pool"""
    try:
        from similar_certificates.similarity import similarity_checker
        similarity_checker(state)
        return state
    except Exception as e:
        logger.error(f"Similarity check failed: {e}")
        return state

def ocr_llm(state):
    """Perform OCR on certificates with timeout"""
    processing_id = state.get("processing_id", "unknown")
    
    try:
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, "Performing OCR on certificates...")
        
        # Add timeout for OCR processing
        future = executor.submit(run_ocr_check, state)
        result_state = future.result(timeout=300)  # 5 minute timeout for OCR
        
        return result_state
        
    except Exception as e:
        error_msg = f"Error in OCR processing: {str(e)}"
        logger.error(error_msg)
        state["messages"].append(error_msg)
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, "OCR failed, continuing...")
        return state

def run_ocr_check(state):
    """Run OCR check using lazy-loaded reader"""
    try:
        ocr_results = {}
        processed_folder = "./processed_certificates"
        
        # Get OCR reader (will initialize if needed)
        reader = get_ocr_reader()

        for file in state["accepted_certi"]:
            if file.endswith(".png"):
                img_path = os.path.join(processed_folder, file)
                result = reader.readtext(img_path, detail=0)
                ocr_results[file] = " ".join(result)

        state["ocr_texts"] = ocr_results
        return state
        
    except Exception as e:
        logger.error(f"OCR check failed: {e}")
        return state

def validation_llm(state):
    """Validate certificates against database with timeout"""
    processing_id = state.get("processing_id", "unknown")
    
    try:
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, "Validating certificates against database...")
        
        # Add timeout for validation
        future = executor.submit(run_validation, state)
        result_state = future.result(timeout=120)  # 2 minute timeout
        
        return result_state
        
    except Exception as e:
        error_msg = f"Error in validation: {str(e)}"
        logger.error(error_msg)
        state["messages"].append(error_msg)
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, "Validation failed, continuing...")
        return state

def run_validation(state):
    """Run validation check - runs in thread pool"""
    try:
        from database import fetch_data
        from langchain_core.prompts import ChatPromptTemplate
        
        for certi, ocr_text in state["ocr_texts"].items():
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an assistant that extracts structured fields from OCR text of certificates.
                Return only valid JSON with fields: EnrollmentNo, Name, Course, CGPA"""),
                ("user", f"OCR Text: {ocr_text}")
            ])
            
            chain = prompt | llm
            response = chain.invoke({})
            try:
                ocr_data = json.loads(response.content)  
            except:
                ocr_data = {"EnrollmentNo": None, "Name": None, "Course": None, "CGPA": None}

            enrollmentNo = ocr_data.get("EnrollmentNo")
            db_record = fetch_data(str(enrollmentNo)) if enrollmentNo else None
            
            if db_record:
                db_record_copy = copy.deepcopy(db_record)
                if "_id" in db_record_copy:
                    db_record_copy["_id"] = str(db_record_copy["_id"])
                    
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a checking ai, check whether the data in the given 
                     certificate and the data in database are same or not. 
                     If same then return true, else return false. 
                     Have strict checking for key sections like enrollment number,
                     but you can be lenient for name and other non-important sections."""),
                    ("user", "database data: {db_record_copy}, certificate data: {ocr_data}")
                ])
                
                chain = prompt | llm
                output = chain.invoke({
                    "db_record_copy": json.dumps(db_record_copy),
                    "ocr_data": json.dumps(ocr_data)
                })
                text = output.content.lower()
                
                if "true" in text:
                    if certi not in state["accepted_certi"]:
                        state["accepted_certi"].append(certi)
                    if certi in state["rejected_certi"]:
                        state["rejected_certi"].remove(certi)
                else:
                    if certi in state["accepted_certi"]:
                        state["accepted_certi"].remove(certi)
                    if certi not in state["rejected_certi"]:
                        state["rejected_certi"].append(certi)
            else:
                if certi in state["accepted_certi"]:
                    state["accepted_certi"].remove(certi)
                if certi not in state["rejected_certi"]:
                    state["rejected_certi"].append(certi)
        
        return state
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return state

def selector_llm(state):
    """Move certificates to appropriate folders"""
    processing_id = state.get("processing_id", "unknown")
    
    try:
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, "Moving certificates to final folders...")
        
        base_dir = get_base_dir()
        curr_path = base_dir / "processed_certificates"
        accepted_path = base_dir / "accepted_certificates"
        rejected_path = base_dir / "rejected_certificates"

        accepted_path.mkdir(exist_ok=True)
        rejected_path.mkdir(exist_ok=True)

        for certi in state["accepted_certi"]:
            src = curr_path / certi
            dst = accepted_path / certi
            if src.exists():
                with Image.open(src) as img:
                    img.save(dst, 'PNG')

        for certi in state["rejected_certi"]:
            src = curr_path / certi
            dst = rejected_path / certi
            if src.exists():
                with Image.open(src) as img:
                    img.save(dst, 'PNG')

        return state
        
    except Exception as e:
        error_msg = f"Error in selector: {str(e)}"
        logger.error(error_msg)
        state["messages"].append(error_msg)
        return state

def clear_certificates_folder():
    """Clear the certificates folder before processing new files"""
    try:
        base_dir = get_base_dir()
        certificates_folder = base_dir / "certificates"
        
        if certificates_folder.exists():
            for file_path in certificates_folder.iterdir():
                if not file_path.name.startswith('.'):  # Keep hidden files
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logger.error(f"Error deleting {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error clearing certificates folder: {e}")

def cleanup_processed_folders():
    """Clean up processed folders after processing"""
    try:
        base_dir = get_base_dir()
        folders_to_clean = [base_dir / "processed_certificates"]
        
        for folder in folders_to_clean:
            if folder.exists():
                for file_path in folder.iterdir():
                    if not file_path.name.startswith('.'):
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                        except Exception as e:
                            logger.error(f"Error deleting processed file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning up processed folders: {e}")

def process_certificates_background(files_data: list, processing_id: str):
    """Background task for processing certificates"""
    try:
        update_processing_status(processing_id, ProcessingStatus.PROCESSING, "Starting certificate processing...")
        
        # Initialize heavy components if not done yet
        if not components_loaded:
            logger.info("Initializing heavy components in background...")
            initialize_heavy_components()
        
        # Save files
        base_dir = get_base_dir()
        certificates_folder = base_dir / "certificates"
        certificates_folder.mkdir(exist_ok=True)
        
        saved_files = []
        for filename, content in files_data:
            file_path = certificates_folder / filename
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            saved_files.append(filename)
        
        # Run pipeline
        initial_state = {
            "messages": [], 
            "ecerti": [], 
            "human": [], 
            "rejected_certi": [], 
            "accepted_certi": [], 
            "ocr_texts": {},
            "processing_id": processing_id
        }
        
        final_state = graph.invoke(initial_state)
        
        # Process messages
        processing_messages = []
        for msg in final_state["messages"]:
            if isinstance(msg, str):
                processing_messages.append(msg)
            elif hasattr(msg, 'content'):
                processing_messages.append(msg.content)
            elif isinstance(msg, dict) and "content" in msg:
                processing_messages.append(msg["content"])
            else:
                processing_messages.append(str(msg))
        
        result_data = {
            "uploaded_files": saved_files,
            "total_files_processed": len(saved_files),
            "results": {
                "accepted_certificates": final_state["accepted_certi"],
                "rejected_certificates": final_state["rejected_certi"],
                "total_accepted": len(final_state["accepted_certi"]),
                "total_rejected": len(final_state["rejected_certi"]),
            },
            "classification": {
                "human_certificates": final_state["human"],
                "e_certificates": final_state["ecerti"]
            },
            "ocr_results": final_state["ocr_texts"],
            "processing_messages": processing_messages
        }
        
        update_processing_status(processing_id, ProcessingStatus.COMPLETED, "Processing completed successfully", result_data)
        
        # Cleanup
        cleanup_processed_folders()
        
    except Exception as e:
        error_msg = f"Background processing failed: {str(e)}"
        logger.error(error_msg)
        update_processing_status(processing_id, ProcessingStatus.FAILED, error_msg)

@app.on_event("startup")
async def startup_event():
    """Initialize only essential components on startup"""
    logger.info("Starting up Certificate Validation API...")
    ensure_directories()
    
    # Only initialize LLM components during startup for fast boot
    success = initialize_llm_components()
    if not success:
        logger.error("Failed to initialize LLM components")
    else:
        logger.info("Essential components initialized - service ready")
        logger.info("Heavy components (OCR, etc.) will be initialized on first request")

@app.get("/")
async def root():
    return {
        "message": "Certificate Validation API", 
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "components": {
            "llm": llm is not None,
            "image_llm": image_llm is not None,
            "certificate_detector": certificate_detector is not None,
            "graph": graph is not None,
            "ocr_reader": ocr_reader is not None,
            "heavy_components_loaded": components_loaded
        }
    }
    return health_status

@app.post("/validate-certificates/")
async def validate_certificates(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload and validate certificates asynchronously"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if not graph:
        raise HTTPException(status_code=503, detail="Service not properly initialized")
    
    try:
        # Generate processing ID
        processing_id = f"proc_{int(time.time())}_{len(files)}"
        
        # Read files into memory
        files_data = []
        for file in files:
            if file.filename:
                content = await file.read()
                files_data.append((file.filename, content))
        
        if not files_data:
            raise HTTPException(status_code=400, detail="No valid files were uploaded")
        
        # Start background processing
        update_processing_status(processing_id, ProcessingStatus.QUEUED, f"Queued {len(files_data)} files for processing")
        background_tasks.add_task(process_certificates_background, files_data, processing_id)
        
        return JSONResponse(content={
            "status": "accepted",
            "processing_id": processing_id,
            "message": f"Processing started for {len(files_data)} files",
            "check_status_url": f"/status/{processing_id}",
            "note": "Heavy components (OCR) will be initialized during processing if needed"
        }, status_code=202)
        
    except Exception as e:
        logger.error(f"Error starting certificate validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

@app.get("/status/{processing_id}")
async def get_processing_status(processing_id: str):
    """Get processing status"""
    if processing_id not in processing_status:
        raise HTTPException(status_code=404, detail="Processing ID not found")
    
    return processing_status[processing_id]

@app.post("/warm-up/")
async def warm_up():
    """Warm up heavy components"""
    global components_loaded
    
    if components_loaded:
        return {"status": "already_loaded", "message": "Heavy components already loaded"}
    
    try:
        # Run in background
        def warm_up_task():
            initialize_heavy_components()
        
        executor.submit(warm_up_task)
        
        return {"status": "started", "message": "Heavy component initialization started in background"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warm-up failed: {str(e)}")

@app.delete("/cleanup/")
async def cleanup_all():
    """Clean up all processing folders"""
    try:
        base_dir = get_base_dir()
        folders_to_clean = [
            "certificates", 
            "processed_certificates", 
            "accepted_certificates", 
            "rejected_certificates"
        ]
        
        cleaned_folders = []
        for folder_name in folders_to_clean:
            folder = base_dir / folder_name
            if folder.exists():
                for file_path in folder.iterdir():
                    if not file_path.name.startswith('.'):
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                        except Exception as e:
                            logger.error(f"Error deleting {file_path}: {e}")
                cleaned_folders.append(folder_name)
        
        return {"status": "success", "cleaned_folders": cleaned_folders}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))