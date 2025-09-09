from typing_extensions import Annotated
from typing import TypedDict
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import base64
import os
import json
from dotenv import load_dotenv
from document_processor.doc_processor import DocumentProcessor
from similar_certificates.similarity import similarity_checker
from certificate_detection.detector import CertificateDetector  
from ocr_checking.ocr import ocr_checker
from database import fetch_data
import copy
from PIL import Image
import io

load_dotenv()

certificate_detector = CertificateDetector()

class State(TypedDict):
    messages : Annotated[list, add_messages]
    human : list
    ecerti : list
    rejected_certi : list
    accepted_certi : list
    ocr_texts : dict

llm = ChatGroq(model="openai/gpt-oss-120b")
image_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")

def resize_image_for_api(image_path, max_size=(1024, 1024), quality=85):
    """
    Resize and compress image to reduce file size for API calls
    """
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (for PNG with transparency)
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


graph_builder = StateGraph(State)


def certificate_type_llm(state: State):
    """
    Step 1: Process documents and convert all images to PNG
    Step 2: Classify all PNG files 
    Step 3: Perform object detection on human-clicked images
    Step 4: Replace original files with cropped certificates
    """
    
    # Step 1: Process documents normally (converts PDFs and images to PNG)
    print("Step 1: Processing documents and converting to PNG...")
    processor = DocumentProcessor()
    processed_image_path = processor.process_documents()
    
    processed_folder = "./processed_certificates"
    if not os.path.exists(processed_folder):
        error_msg = "Processed certificates folder not found."
        state["messages"].append({"role": "assistant", "content": error_msg})
        return state
    
    png_files = [f for f in os.listdir(processed_folder) if f.lower().endswith('.png')]
    
    if not png_files:
        error_msg = "No processed PNG files found."
        state["messages"].append({"role": "assistant", "content": error_msg})
        return state
    
    print(f"Found {len(png_files)} processed certificate(s) to classify")
    
    # Step 2: Classify all PNG files first
    print("Step 2: Classifying all certificates...")
    classified_human = []
    classified_ecerti = []
    
    for png_file in png_files:
        image_path = os.path.join(processed_folder, png_file)
        
        print(f"Classifying: {png_file}")
        
        try:
            compressed_image_data = resize_image_for_api(image_path)
            img_b64 = base64.b64encode(compressed_image_data).decode("utf-8")
            
            if len(img_b64) > 4_000_000:
                print(f"Warning: {png_file} is still large after compression")
                compressed_image_data = resize_image_for_api(image_path, max_size=(512, 512), quality=60)
                img_b64 = base64.b64encode(compressed_image_data).decode("utf-8")
            
            # Use LLM classification for all images
            prompt = [
                {"role": "system", "content": "You are an assistant that classifies certificate type. Don't give me any extra information, just tell me whether the certificate is ecertificate or normal human clicked image of the certificate"},
                {"role": "user", "content": [
                    {"type": "text", "text": "Classify this certificate:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]}
            ]

            response = image_llm.invoke(prompt)
            classification = response.content
            
            if("ecertificate" in classification.lower()):
                classified_ecerti.append(png_file)
                print(f"Certificate {png_file} classified as e-certificate")
            else:
                classified_human.append(png_file)
                print(f"Certificate {png_file} classified as human-clicked")
            
            state["messages"].append({
                "role": "assistant", 
                "content": f"Certificate: {png_file} | Classification: {classification}"
            })
            
        except Exception as e:
            error_msg = f"Error processing {png_file}: {str(e)}"
            print(error_msg)
            state["messages"].append({
                "role": "assistant", 
                "content": error_msg
            })
            continue
    
    # Step 3: Perform object detection on human-clicked images
    print("Step 3: Performing object detection on human-clicked images...")
    final_human_certificates = []
    
    for human_cert in classified_human:
        image_path = os.path.join(processed_folder, human_cert)
        
        print(f"Processing {human_cert} for object detection...")
        
        # Use object detection to crop certificates
        cropped_paths = certificate_detector.detect_and_crop_certificates(
            image_path=image_path,
            output_folder=processed_folder
        )
        
        if cropped_paths:
            print(f"Object detection successful for {human_cert}, found {len(cropped_paths)} certificates")
            # Remove the original file since we have cropped versions
            original_path = os.path.join(processed_folder, human_cert)
            if os.path.exists(original_path):
                os.remove(original_path)
                print(f"Removed original file: {human_cert}")
            
            # Add cropped certificates to final list
            final_human_certificates.extend(cropped_paths)
        else:
            print(f"No certificates detected in {human_cert}, keeping original")
            # If no detection, keep the original file
            final_human_certificates.append(human_cert)
    
    # Step 4: Update state with final classifications
    state["human"] = final_human_certificates
    state["ecerti"] = classified_ecerti
    
    print(f"Final classification - Human: {len(final_human_certificates)}, E-certificates: {len(classified_ecerti)}")
    print(f"Human certificates: {final_human_certificates}")
    print(f"E-certificates: {classified_ecerti}")
    
    return state




def similarity_checking_llm(state:State) :
    similarity_checker(state)
    return state

def ocr_llm(state:State) :
    ocr_checker(state)
    return state

def validation_llm(state:State) :
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
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """
                     you are a checking ai, check whether the data in the given 
                     in certificate and the data in database are same or not. 
                     If same then return true, else return false. 
                     Have a strict checking for key sections like enrollment number and all,
                     but you can keep a little linient checking for name and other non important sections, but still they muct be very much similar"""),
                     ("user", "database data:  {db_record_copy}, certificate data : {ocr_data}")
                ]
            )
            chain = prompt|llm
            output = chain.invoke({
                "db_record_copy" : json.dumps(db_record_copy),
                "ocr_data" : json.dumps(ocr_data)
            })
            text = output.content.lower()
            
            # match = (
            #     str(db_record.get("enrollmentNo")) == str(ocr_data.get("EnrollmentNo")) and
            #     str(db_record.get("name")).lower() == str(ocr_data.get("Name")).lower() and
            #     str(db_record.get("course")).lower() == str(ocr_data.get("Course")).lower() and
            #     float(db_record.get("cgpa")) == float(ocr_data.get("CGPA"))
            # )
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

def selector_llm(state: State):
    curr_path = "./processed_certificates"
    accepted_path = "./accepted_certificates"
    rejected_path = "./rejected_certificates"

    os.makedirs(accepted_path, exist_ok=True)
    os.makedirs(rejected_path, exist_ok=True)

    for certi in state["accepted_certi"]:
        src = os.path.join(curr_path, certi)
        dst = os.path.join(accepted_path, certi)
        if os.path.exists(src):
            with Image.open(src) as img:
                img.save(dst, 'PNG')

    for certi in state["rejected_certi"]:
        src = os.path.join(curr_path, certi)
        dst = os.path.join(rejected_path, certi)
        if os.path.exists(src):
            with Image.open(src) as img:
                img.save(dst, 'PNG')

    return state

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

final_state = graph.invoke({"messages": [], "ecerti": [], "human": [], "rejected_certi":[], "accepted_certi":[], "ocr_texts" : {}})

print("Accepted Cerificate", final_state["accepted_certi"])
print("Rejected Certificates", final_state["rejected_certi"])
print("OCR Texts:", final_state["ocr_texts"])
