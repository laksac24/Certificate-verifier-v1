# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List
# import os
# import shutil
# import zipfile
# import uuid
# from pathlib import Path
# import asyncio
# import logging

# # Import your existing processing logic
# from app import graph  # Your updated graph from app.py

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Certificate Verification API")

# # Enable CORS for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure properly for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Simple in-memory storage for job status
# processing_jobs = {}

# def clear_certificates_folder():
#     """Clear the certificates folder before new uploads"""
#     cert_folder = "./certificates"
#     if os.path.exists(cert_folder):
#         for file in os.listdir(cert_folder):
#             file_path = os.path.join(cert_folder, file)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#     else:
#         os.makedirs(cert_folder, exist_ok=True)
#     logger.info("Cleared certificates folder")

# def clear_result_folders():
#     """Clear previous results"""
#     folders = ["./processed_certificates", "./accepted_certificates", "./rejected_certificates"]
#     for folder in folders:
#         if os.path.exists(folder):
#             shutil.rmtree(folder)
#         os.makedirs(folder, exist_ok=True)
#     logger.info("Cleared result folders")

# def create_zip_file(folder_path: str, zip_name: str) -> str:
#     """Create zip file from folder contents"""
#     zip_path = f"./{zip_name}"
#     with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         if os.path.exists(folder_path):
#             files_added = 0
#             for file in os.listdir(folder_path):
#                 file_path = os.path.join(folder_path, file)
#                 if os.path.isfile(file_path):
#                     zipf.write(file_path, file)
#                     files_added += 1
#             logger.info(f"Created zip {zip_name} with {files_added} files")
#     return zip_path

# @app.get("/")
# async def root():
#     return {
#         "message": "Certificate Verification API is running!",
#         "version": "2.0",
#         "features": [
#             "Automatic document processing",
#             "Certificate classification",
#             "Object detection for human-clicked images",
#             "OCR and database validation",
#             "Similarity checking"
#         ]
#     }

# @app.post("/upload-and-process")
# async def upload_and_process_certificates(files: List[UploadFile] = File(...)):
#     """Upload certificates and process them with the updated workflow"""
    
#     if not files or len(files) == 0:
#         raise HTTPException(status_code=400, detail="No files uploaded")
    
#     # Generate job ID
#     job_id = str(uuid.uuid4())
#     processing_jobs[job_id] = {
#         "status": "processing", 
#         "message": "Starting certificate processing...",
#         "progress": "Uploading files"
#     }
    
#     try:
#         logger.info(f"Starting job {job_id} with {len(files)} files")
        
#         # Clear previous files
#         clear_certificates_folder()
#         clear_result_folders()
        
#         # Update status
#         processing_jobs[job_id]["progress"] = "Saving uploaded files"
        
#         # Save uploaded files to certificates folder
#         saved_files = []
#         for file in files:
#             # Validate file type
#             allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']
#             if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
#                 raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Allowed: {', '.join(allowed_extensions)}")
            
#             file_path = f"./certificates/{file.filename}"
#             with open(file_path, "wb") as buffer:
#                 content = await file.read()
#                 buffer.write(content)
#             saved_files.append(file.filename)
        
#         logger.info(f"Saved {len(saved_files)} files: {saved_files}")
        
#         # Update status
#         processing_jobs[job_id].update({
#             "message": "Files uploaded successfully, starting processing...",
#             "progress": "Converting documents to PNG",
#             "uploaded_files": saved_files
#         })
        
#         # Run your existing processing pipeline with updated workflow
#         logger.info("Starting processing pipeline...")
#         processing_jobs[job_id]["progress"] = "Processing certificates (Step 1: Document conversion)"
        
#         final_state = graph.invoke({
#             "messages": [], 
#             "ecerti": [], 
#             "human": [], 
#             "rejected_certi": [], 
#             "accepted_certi": [], 
#             "ocr_texts": {}
#         })
        
#         logger.info("Processing completed")
#         logger.info(f"Results - Accepted: {len(final_state['accepted_certi'])}, Rejected: {len(final_state['rejected_certi'])}")
#         logger.info(f"E-certificates: {len(final_state['ecerti'])}, Human-clicked: {len(final_state['human'])}")
        
#         # Update status
#         processing_jobs[job_id]["progress"] = "Creating result packages"
        
#         # Create zip files for results
#         accepted_zip = None
#         rejected_zip = None
        
#         if final_state["accepted_certi"]:
#             accepted_zip = f"accepted_{job_id}.zip"
#             create_zip_file("./accepted_certificates", accepted_zip)
        
#         if final_state["rejected_certi"]:
#             rejected_zip = f"rejected_{job_id}.zip"
#             create_zip_file("./rejected_certificates", rejected_zip)
        
#         # Count processed files
#         total_processed = len(final_state["accepted_certi"]) + len(final_state["rejected_certi"])
        
#         # Update job status
#         processing_jobs[job_id] = {
#             "status": "completed",
#             "message": "Processing completed successfully",
#             "progress": "Completed",
#             "uploaded_files": saved_files,
#             "processing_details": {
#                 "workflow_steps": [
#                     "Document conversion to PNG",
#                     "Certificate classification (LLM)",
#                     "Object detection on human-clicked images",
#                     "Certificate cropping and replacement",
#                     "Similarity checking",
#                     "OCR extraction",
#                     "Database validation"
#                 ]
#             },
#             "results": {
#                 "accepted_certificates": final_state["accepted_certi"],
#                 "rejected_certificates": final_state["rejected_certi"],
#                 "e_certificates": final_state["ecerti"],
#                 "human_certificates": final_state["human"],
#                 "total_processed": total_processed,
#                 "accepted_zip": accepted_zip,
#                 "rejected_zip": rejected_zip,
#                 "ocr_texts": final_state["ocr_texts"]
#             }
#         }
        
#         return {
#             "job_id": job_id,
#             "status": "completed",
#             "message": "Processing completed successfully",
#             "summary": {
#                 "total_uploaded": len(saved_files),
#                 "total_processed": total_processed,
#                 "accepted_count": len(final_state["accepted_certi"]),
#                 "rejected_count": len(final_state["rejected_certi"]),
#                 "e_certificates_count": len(final_state["ecerti"]),
#                 "human_certificates_count": len(final_state["human"]),
#                 "object_detection_applied": len([cert for cert in final_state["human"] if "detected" in cert])
#             },
#             "processing_details": {
#                 "workflow": "PNG Conversion → Classification → Object Detection → Similarity Check → OCR → Validation",
#                 "object_detection_used": len(final_state["human"]) > 0,
#                 "certificates_cropped": len([cert for cert in final_state["human"] if "detected" in cert])
#             },
#             "download_links": {
#                 "accepted": f"/download/accepted/{job_id}" if accepted_zip else None,
#                 "rejected": f"/download/rejected/{job_id}" if rejected_zip else None
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"Processing failed for job {job_id}: {str(e)}")
#         processing_jobs[job_id] = {
#             "status": "failed", 
#             "message": f"Processing failed: {str(e)}",
#             "progress": "Failed",
#             "error_details": str(e)
#         }
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# @app.get("/download/accepted/{job_id}")
# async def download_accepted_certificates(job_id: str):
#     """Download accepted certificates as zip file"""
    
#     if job_id not in processing_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     job = processing_jobs[job_id]
#     if job["status"] != "completed":
#         raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    
#     zip_file = job["results"].get("accepted_zip")
#     if not zip_file or not os.path.exists(zip_file):
#         raise HTTPException(status_code=404, detail="No accepted certificates found")
    
#     logger.info(f"Downloading accepted certificates for job {job_id}")
    
#     return FileResponse(
#         path=zip_file,
#         filename=f"accepted_certificates_{job_id}.zip",
#         media_type='application/zip'
#     )

# @app.get("/download/rejected/{job_id}")
# async def download_rejected_certificates(job_id: str):
#     """Download rejected certificates as zip file"""
    
#     if job_id not in processing_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     job = processing_jobs[job_id]
#     if job["status"] != "completed":
#         raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    
#     zip_file = job["results"].get("rejected_zip")
#     if not zip_file or not os.path.exists(zip_file):
#         raise HTTPException(status_code=404, detail="No rejected certificates found")
    
#     logger.info(f"Downloading rejected certificates for job {job_id}")
    
#     return FileResponse(
#         path=zip_file,
#         filename=f"rejected_certificates_{job_id}.zip",
#         media_type='application/zip'
#     )

# @app.get("/status/{job_id}")
# async def get_job_status(job_id: str):
#     """Get detailed processing status"""
    
#     if job_id not in processing_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     return processing_jobs[job_id]

# @app.get("/jobs")
# async def list_all_jobs():
#     """List all jobs and their statuses"""
#     return {
#         "total_jobs": len(processing_jobs),
#         "jobs": {job_id: {"status": job["status"], "message": job["message"]} 
#                 for job_id, job in processing_jobs.items()}
#     }

# @app.delete("/cleanup/{job_id}")
# async def cleanup_job_files(job_id: str):
#     """Clean up temporary files for a job"""
    
#     if job_id not in processing_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     # Remove zip files
#     job = processing_jobs[job_id]
#     if job["status"] == "completed" and "results" in job:
#         results = job["results"]
#         for zip_type in ["accepted_zip", "rejected_zip"]:
#             zip_file = results.get(zip_type)
#             if zip_file and os.path.exists(zip_file):
#                 os.remove(zip_file)
#                 logger.info(f"Removed {zip_file}")
    
#     # Remove job from memory
#     del processing_jobs[job_id]
#     logger.info(f"Cleaned up job {job_id}")
    
#     return {"message": "Job files cleaned up successfully"}

# @app.delete("/cleanup-all")
# async def cleanup_all_jobs():
#     """Clean up all job files"""
    
#     cleanup_count = 0
#     for job_id in list(processing_jobs.keys()):
#         job = processing_jobs[job_id]
#         if job["status"] == "completed" and "results" in job:
#             results = job["results"]
#             for zip_type in ["accepted_zip", "rejected_zip"]:
#                 zip_file = results.get(zip_type)
#                 if zip_file and os.path.exists(zip_file):
#                     os.remove(zip_file)
#         cleanup_count += 1
    
#     processing_jobs.clear()
#     logger.info(f"Cleaned up {cleanup_count} jobs")
    
#     return {"message": f"Cleaned up {cleanup_count} jobs successfully"}

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "active_jobs": len([job for job in processing_jobs.values() if job["status"] == "processing"]),
#         "completed_jobs": len([job for job in processing_jobs.values() if job["status"] == "completed"]),
#         "failed_jobs": len([job for job in processing_jobs.values() if job["status"] == "failed"])
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")



from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil
import zipfile
import uuid
from pathlib import Path
import asyncio
import logging
import tempfile
import io
from contextlib import contextmanager

from app import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Certificate Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processing_jobs = {}

@contextmanager
def temporary_processing_environment():
    temp_dir = tempfile.mkdtemp(prefix="cert_processing_")
    
    cert_folder = os.path.join(temp_dir, "certificates")
    processed_folder = os.path.join(temp_dir, "processed_certificates")
    accepted_folder = os.path.join(temp_dir, "accepted_certificates")
    rejected_folder = os.path.join(temp_dir, "rejected_certificates")
    
    for folder in [cert_folder, processed_folder, accepted_folder, rejected_folder]:
        os.makedirs(folder, exist_ok=True)
    
    original_cwd = os.getcwd()
    
    try:
        temp_links = []
        for folder_name in ["certificates", "processed_certificates", "accepted_certificates", "rejected_certificates"]:
            link_path = os.path.join(original_cwd, folder_name)
            target_path = os.path.join(temp_dir, folder_name)
            
            if os.path.exists(link_path):
                if os.path.islink(link_path):
                    os.unlink(link_path)
                elif os.path.isdir(link_path):
                    shutil.rmtree(link_path)
            
            os.symlink(target_path, link_path)
            temp_links.append(link_path)
        
        yield {
            'temp_dir': temp_dir,
            'cert_folder': cert_folder,
            'accepted_folder': accepted_folder,
            'rejected_folder': rejected_folder,
            'temp_links': temp_links
        }
        
    finally:
        for link_path in temp_links:
            if os.path.exists(link_path):
                os.unlink(link_path)
        
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

def create_in_memory_zip(folder_path: str) -> io.BytesIO:
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists(folder_path):
            files_added = 0
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    zipf.write(file_path, file)
                    files_added += 1
            logger.info(f"Created in-memory zip with {files_added} files")
    
    zip_buffer.seek(0)
    return zip_buffer

@app.get("/")
async def root():
    return {
        "message": "Certificate Verification API is running!",
        "version": "2.1",
        "storage_policy": "Temporary processing only - no files stored permanently"
    }

@app.post("/upload-and-process")
async def upload_and_process_certificates(files: List[UploadFile] = File(...)):
    
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        "status": "processing", 
        "message": "Starting certificate processing...",
        "progress": "Uploading files"
    }
    
    try:
        logger.info(f"Starting job {job_id} with {len(files)} files")
        
        with temporary_processing_environment() as temp_env:
            
            processing_jobs[job_id]["progress"] = "Saving uploaded files to temporary storage"
            
            saved_files = []
            for file in files:
                allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']
                if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
                    raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Allowed: {', '.join(allowed_extensions)}")
                
                file_path = os.path.join(temp_env['cert_folder'], file.filename)
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                saved_files.append(file.filename)
            
            logger.info(f"Saved {len(saved_files)} files to temporary storage: {saved_files}")
            
            processing_jobs[job_id].update({
                "message": "Files uploaded to temporary storage, starting processing...",
                "progress": "Converting documents to PNG",
                "uploaded_files": saved_files
            })
            
            logger.info("Starting processing pipeline...")
            processing_jobs[job_id]["progress"] = "Processing certificates"
            
            final_state = graph.invoke({
                "messages": [], 
                "ecerti": [], 
                "human": [], 
                "rejected_certi": [], 
                "accepted_certi": [], 
                "ocr_texts": {}
            })
            
            logger.info("Processing completed")
            logger.info(f"Results - Accepted: {len(final_state['accepted_certi'])}, Rejected: {len(final_state['rejected_certi'])}")
            
            processing_jobs[job_id]["progress"] = "Creating result packages in memory"
            
            accepted_zip_data = None
            rejected_zip_data = None
            
            if final_state["accepted_certi"]:
                accepted_zip_data = create_in_memory_zip(temp_env['accepted_folder'])
            
            if final_state["rejected_certi"]:
                rejected_zip_data = create_in_memory_zip(temp_env['rejected_folder'])
            
            total_processed = len(final_state["accepted_certi"]) + len(final_state["rejected_certi"])
            
            processing_jobs[job_id] = {
                "status": "completed",
                "message": "Processing completed successfully",
                "progress": "Completed",
                "uploaded_files": saved_files,
                "results": {
                    "accepted_certificates": final_state["accepted_certi"],
                    "rejected_certificates": final_state["rejected_certi"],
                    "e_certificates": final_state["ecerti"],
                    "human_certificates": final_state["human"],
                    "total_processed": total_processed,
                    "accepted_zip_data": accepted_zip_data,
                    "rejected_zip_data": rejected_zip_data,
                    "ocr_texts": final_state["ocr_texts"]
                }
            }
        
        logger.info(f"Temporary processing environment cleaned up for job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "message": "Processing completed successfully",
            "summary": {
                "total_uploaded": len(saved_files),
                "total_processed": total_processed,
                "accepted_count": len(final_state["accepted_certi"]),
                "rejected_count": len(final_state["rejected_certi"]),
                "e_certificates_count": len(final_state["ecerti"]),
                "human_certificates_count": len(final_state["human"])
            },
            "download_links": {
                "accepted": f"/download/accepted/{job_id}" if accepted_zip_data else None,
                "rejected": f"/download/rejected/{job_id}" if rejected_zip_data else None
            }
        }
        
    except Exception as e:
        logger.error(f"Processing failed for job {job_id}: {str(e)}")
        processing_jobs[job_id] = {
            "status": "failed", 
            "message": f"Processing failed: {str(e)}",
            "progress": "Failed",
            "error_details": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/download/accepted/{job_id}")
async def download_accepted_certificates(job_id: str):
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    
    zip_data = job["results"].get("accepted_zip_data")
    if not zip_data:
        raise HTTPException(status_code=404, detail="No accepted certificates found")
    
    logger.info(f"Streaming accepted certificates for job {job_id}")
    
    zip_buffer = io.BytesIO(zip_data.getvalue())
    zip_buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(zip_buffer.read()),
        media_type='application/zip',
        headers={"Content-Disposition": f"attachment; filename=accepted_certificates_{job_id}.zip"}
    )

@app.get("/download/rejected/{job_id}")
async def download_rejected_certificates(job_id: str):
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    
    zip_data = job["results"].get("rejected_zip_data")
    if not zip_data:
        raise HTTPException(status_code=404, detail="No rejected certificates found")
    
    logger.info(f"Streaming rejected certificates for job {job_id}")
    
    zip_buffer = io.BytesIO(zip_data.getvalue())
    zip_buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(zip_buffer.read()),
        media_type='application/zip',
        headers={"Content-Disposition": f"attachment; filename=rejected_certificates_{job_id}.zip"}
    )

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = processing_jobs[job_id].copy()
    if "results" in status and "accepted_zip_data" in status["results"]:
        status["results"] = {k: v for k, v in status["results"].items() 
                           if k not in ["accepted_zip_data", "rejected_zip_data"]}
        status["results"]["download_ready"] = True
    
    return status

@app.get("/jobs")
async def list_all_jobs():
    return {
        "total_jobs": len(processing_jobs),
        "jobs": {job_id: {
            "status": job["status"], 
            "message": job["message"],
            "has_downloads": job.get("results", {}).get("accepted_zip_data") is not None or 
                           job.get("results", {}).get("rejected_zip_data") is not None
        } for job_id, job in processing_jobs.items()}
    }

@app.delete("/cleanup/{job_id}")
async def cleanup_job_files(job_id: str):
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del processing_jobs[job_id]
    logger.info(f"Cleaned up job {job_id} from memory")
    
    return {"message": "Job cleaned up successfully from memory"}

@app.delete("/cleanup-all")
async def cleanup_all_jobs():
    
    cleanup_count = len(processing_jobs)
    processing_jobs.clear()
    logger.info(f"Cleaned up {cleanup_count} jobs from memory")
    
    return {"message": f"Cleaned up {cleanup_count} jobs successfully from memory"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_jobs": len([job for job in processing_jobs.values() if job["status"] == "processing"]),
        "completed_jobs": len([job for job in processing_jobs.values() if job["status"] == "completed"]),
        "failed_jobs": len([job for job in processing_jobs.values() if job["status"] == "failed"])
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")