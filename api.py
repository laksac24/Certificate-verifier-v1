# # from fastapi import FastAPI, File, UploadFile, HTTPException
# # from fastapi.responses import StreamingResponse
# # from fastapi.middleware.cors import CORSMiddleware
# # from typing import List
# # import os
# # import shutil
# # import zipfile
# # import uuid
# # from pathlib import Path
# # import asyncio
# # import logging
# # import tempfile
# # import io
# # from contextlib import contextmanager

# # from app import graph

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # app = FastAPI(title="Certificate Verification API")

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # processing_jobs = {}

# # @contextmanager
# # def temporary_processing_environment():
# #     temp_dir = tempfile.mkdtemp(prefix="cert_processing_")
    
# #     cert_folder = os.path.join(temp_dir, "certificates")
# #     processed_folder = os.path.join(temp_dir, "processed_certificates")
# #     accepted_folder = os.path.join(temp_dir, "accepted_certificates")
# #     rejected_folder = os.path.join(temp_dir, "rejected_certificates")
    
# #     for folder in [cert_folder, processed_folder, accepted_folder, rejected_folder]:
# #         os.makedirs(folder, exist_ok=True)
    
# #     original_cwd = os.getcwd()
    
# #     try:
# #         temp_links = []
# #         for folder_name in ["certificates", "processed_certificates", "accepted_certificates", "rejected_certificates"]:
# #             link_path = os.path.join(original_cwd, folder_name)
# #             target_path = os.path.join(temp_dir, folder_name)
            
# #             if os.path.exists(link_path):
# #                 if os.path.islink(link_path):
# #                     os.unlink(link_path)
# #                 elif os.path.isdir(link_path):
# #                     shutil.rmtree(link_path)
            
# #             os.symlink(target_path, link_path)
# #             temp_links.append(link_path)
        
# #         yield {
# #             'temp_dir': temp_dir,
# #             'cert_folder': cert_folder,
# #             'accepted_folder': accepted_folder,
# #             'rejected_folder': rejected_folder,
# #             'temp_links': temp_links
# #         }
        
# #     finally:
# #         for link_path in temp_links:
# #             if os.path.exists(link_path):
# #                 os.unlink(link_path)
        
# #         shutil.rmtree(temp_dir)
# #         logger.info(f"Cleaned up temporary directory: {temp_dir}")

# # def create_in_memory_zip(folder_path: str) -> io.BytesIO:
# #     zip_buffer = io.BytesIO()
    
# #     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
# #         if os.path.exists(folder_path):
# #             files_added = 0
# #             for file in os.listdir(folder_path):
# #                 file_path = os.path.join(folder_path, file)
# #                 if os.path.isfile(file_path):
# #                     zipf.write(file_path, file)
# #                     files_added += 1
# #             logger.info(f"Created in-memory zip with {files_added} files")
    
# #     zip_buffer.seek(0)
# #     return zip_buffer

# # @app.get("/")
# # async def root():
# #     return {
# #         "message": "Certificate Verification API is running!",
# #         "version": "2.1",
# #         "storage_policy": "Temporary processing only - no files stored permanently"
# #     }

# # @app.post("/upload-and-process")
# # async def upload_and_process_certificates(files: List[UploadFile] = File(...)):
    
# #     if not files or len(files) == 0:
# #         raise HTTPException(status_code=400, detail="No files uploaded")
    
# #     job_id = str(uuid.uuid4())
# #     processing_jobs[job_id] = {
# #         "status": "processing", 
# #         "message": "Starting certificate processing...",
# #         "progress": "Uploading files"
# #     }
    
# #     try:
# #         logger.info(f"Starting job {job_id} with {len(files)} files")
        
# #         with temporary_processing_environment() as temp_env:
            
# #             processing_jobs[job_id]["progress"] = "Saving uploaded files to temporary storage"
            
# #             saved_files = []
# #             for file in files:
# #                 allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']
# #                 if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
# #                     raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Allowed: {', '.join(allowed_extensions)}")
                
# #                 file_path = os.path.join(temp_env['cert_folder'], file.filename)
# #                 with open(file_path, "wb") as buffer:
# #                     content = await file.read()
# #                     buffer.write(content)
# #                 saved_files.append(file.filename)
            
# #             logger.info(f"Saved {len(saved_files)} files to temporary storage: {saved_files}")
            
# #             processing_jobs[job_id].update({
# #                 "message": "Files uploaded to temporary storage, starting processing...",
# #                 "progress": "Converting documents to PNG",
# #                 "uploaded_files": saved_files
# #             })
            
# #             logger.info("Starting processing pipeline...")
# #             processing_jobs[job_id]["progress"] = "Processing certificates"
            
# #             final_state = graph.invoke({
# #                 "messages": [], 
# #                 "ecerti": [], 
# #                 "human": [], 
# #                 "rejected_certi": [], 
# #                 "accepted_certi": [], 
# #                 "ocr_texts": {}
# #             })
            
# #             logger.info("Processing completed")
# #             logger.info(f"Results - Accepted: {len(final_state['accepted_certi'])}, Rejected: {len(final_state['rejected_certi'])}")
            
# #             processing_jobs[job_id]["progress"] = "Creating result packages in memory"
            
# #             accepted_zip_data = None
# #             rejected_zip_data = None
            
# #             if final_state["accepted_certi"]:
# #                 accepted_zip_data = create_in_memory_zip(temp_env['accepted_folder'])
            
# #             if final_state["rejected_certi"]:
# #                 rejected_zip_data = create_in_memory_zip(temp_env['rejected_folder'])
            
# #             total_processed = len(final_state["accepted_certi"]) + len(final_state["rejected_certi"])
            
# #             processing_jobs[job_id] = {
# #                 "status": "completed",
# #                 "message": "Processing completed successfully",
# #                 "progress": "Completed",
# #                 "uploaded_files": saved_files,
# #                 "results": {
# #                     "accepted_certificates": final_state["accepted_certi"],
# #                     "rejected_certificates": final_state["rejected_certi"],
# #                     "e_certificates": final_state["ecerti"],
# #                     "human_certificates": final_state["human"],
# #                     "total_processed": total_processed,
# #                     "accepted_zip_data": accepted_zip_data,
# #                     "rejected_zip_data": rejected_zip_data,
# #                     "ocr_texts": final_state["ocr_texts"]
# #                 }
# #             }
        
# #         logger.info(f"Temporary processing environment cleaned up for job {job_id}")
        
# #         return {
# #             "job_id": job_id,
# #             "status": "completed",
# #             "message": "Processing completed successfully",
# #             "summary": {
# #                 "total_uploaded": len(saved_files),
# #                 "total_processed": total_processed,
# #                 "accepted_count": len(final_state["accepted_certi"]),
# #                 "rejected_count": len(final_state["rejected_certi"]),
# #                 "e_certificates_count": len(final_state["ecerti"]),
# #                 "human_certificates_count": len(final_state["human"])
# #             },
# #             "download_links": {
# #                 "accepted": f"/download/accepted/{job_id}" if accepted_zip_data else None,
# #                 "rejected": f"/download/rejected/{job_id}" if rejected_zip_data else None
# #             }
# #         }
        
# #     except Exception as e:
# #         logger.error(f"Processing failed for job {job_id}: {str(e)}")
# #         processing_jobs[job_id] = {
# #             "status": "failed", 
# #             "message": f"Processing failed: {str(e)}",
# #             "progress": "Failed",
# #             "error_details": str(e)
# #         }
# #         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# # @app.get("/download/accepted/{job_id}")
# # async def download_accepted_certificates(job_id: str):
    
# #     if job_id not in processing_jobs:
# #         raise HTTPException(status_code=404, detail="Job not found")
    
# #     job = processing_jobs[job_id]
# #     if job["status"] != "completed":
# #         raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    
# #     zip_data = job["results"].get("accepted_zip_data")
# #     if not zip_data:
# #         raise HTTPException(status_code=404, detail="No accepted certificates found")
    
# #     logger.info(f"Streaming accepted certificates for job {job_id}")
    
# #     zip_buffer = io.BytesIO(zip_data.getvalue())
# #     zip_buffer.seek(0)
    
# #     return StreamingResponse(
# #         io.BytesIO(zip_buffer.read()),
# #         media_type='application/zip',
# #         headers={"Content-Disposition": f"attachment; filename=accepted_certificates_{job_id}.zip"}
# #     )

# # @app.get("/download/rejected/{job_id}")
# # async def download_rejected_certificates(job_id: str):
    
# #     if job_id not in processing_jobs:
# #         raise HTTPException(status_code=404, detail="Job not found")
    
# #     job = processing_jobs[job_id]
# #     if job["status"] != "completed":
# #         raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    
# #     zip_data = job["results"].get("rejected_zip_data")
# #     if not zip_data:
# #         raise HTTPException(status_code=404, detail="No rejected certificates found")
    
# #     logger.info(f"Streaming rejected certificates for job {job_id}")
    
# #     zip_buffer = io.BytesIO(zip_data.getvalue())
# #     zip_buffer.seek(0)
    
# #     return StreamingResponse(
# #         io.BytesIO(zip_buffer.read()),
# #         media_type='application/zip',
# #         headers={"Content-Disposition": f"attachment; filename=rejected_certificates_{job_id}.zip"}
# #     )

# # @app.get("/status/{job_id}")
# # async def get_job_status(job_id: str):
    
# #     if job_id not in processing_jobs:
# #         raise HTTPException(status_code=404, detail="Job not found")
    
# #     status = processing_jobs[job_id].copy()
# #     if "results" in status and "accepted_zip_data" in status["results"]:
# #         status["results"] = {k: v for k, v in status["results"].items() 
# #                            if k not in ["accepted_zip_data", "rejected_zip_data"]}
# #         status["results"]["download_ready"] = True
    
# #     return status

# # @app.get("/jobs")
# # async def list_all_jobs():
# #     return {
# #         "total_jobs": len(processing_jobs),
# #         "jobs": {job_id: {
# #             "status": job["status"], 
# #             "message": job["message"],
# #             "has_downloads": job.get("results", {}).get("accepted_zip_data") is not None or 
# #                            job.get("results", {}).get("rejected_zip_data") is not None
# #         } for job_id, job in processing_jobs.items()}
# #     }

# # @app.delete("/cleanup/{job_id}")
# # async def cleanup_job_files(job_id: str):
    
# #     if job_id not in processing_jobs:
# #         raise HTTPException(status_code=404, detail="Job not found")
    
# #     del processing_jobs[job_id]
# #     logger.info(f"Cleaned up job {job_id} from memory")
    
# #     return {"message": "Job cleaned up successfully from memory"}

# # @app.delete("/cleanup-all")
# # async def cleanup_all_jobs():
    
# #     cleanup_count = len(processing_jobs)
# #     processing_jobs.clear()
# #     logger.info(f"Cleaned up {cleanup_count} jobs from memory")
    
# #     return {"message": f"Cleaned up {cleanup_count} jobs successfully from memory"}

# # @app.get("/health")
# # async def health_check():
# #     return {
# #         "status": "healthy",
# #         "active_jobs": len([job for job in processing_jobs.values() if job["status"] == "processing"]),
# #         "completed_jobs": len([job for job in processing_jobs.values() if job["status"] == "completed"]),
# #         "failed_jobs": len([job for job in processing_jobs.values() if job["status"] == "failed"])
# #     }

# # if __name__ == "__main__":
# #     import uvicorn
# #     port = int(os.environ.get("PORT", 4040))
# #     uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")



# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List
# import os
# import shutil
# import zipfile
# import uuid
# from pathlib import Path
# import asyncio
# import logging
# import tempfile
# import io
# import signal
# import sys
# from contextlib import contextmanager

# # Import your graph module
# from app import graph

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Get PORT from environment (Render requirement)
# PORT = int(os.environ.get("PORT", 4040))

# app = FastAPI(
#     title="Certificate Verification API",
#     version="2.1",
#     description="Certificate verification and processing API deployed on Render"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# processing_jobs = {}

# # Graceful shutdown handler for Render
# def signal_handler(sig, frame):
#     logger.info("Shutting down gracefully...")
#     sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)
# signal.signal(signal.SIGTERM, signal_handler)

# @contextmanager
# def temporary_processing_environment():
#     """Create temporary processing environment with proper cleanup"""
#     temp_dir = tempfile.mkdtemp(prefix="cert_processing_")
    
#     cert_folder = os.path.join(temp_dir, "certificates")
#     processed_folder = os.path.join(temp_dir, "processed_certificates")
#     accepted_folder = os.path.join(temp_dir, "accepted_certificates")
#     rejected_folder = os.path.join(temp_dir, "rejected_certificates")
    
#     for folder in [cert_folder, processed_folder, accepted_folder, rejected_folder]:
#         os.makedirs(folder, exist_ok=True)
    
#     original_cwd = os.getcwd()
    
#     try:
#         temp_links = []
#         folder_names = ["certificates", "processed_certificates", "accepted_certificates", "rejected_certificates"]
        
#         for folder_name in folder_names:
#             link_path = os.path.join(original_cwd, folder_name)
#             target_path = os.path.join(temp_dir, folder_name)
            
#             # Clean up existing links/folders
#             if os.path.exists(link_path):
#                 if os.path.islink(link_path):
#                     os.unlink(link_path)
#                 elif os.path.isdir(link_path):
#                     shutil.rmtree(link_path)
            
#             # Create symlink
#             try:
#                 os.symlink(target_path, link_path)
#                 temp_links.append(link_path)
#             except OSError as e:
#                 logger.warning(f"Could not create symlink for {folder_name}: {e}")
#                 # Fallback: use the temp folder directly
#                 temp_links.append(link_path)
        
#         yield {
#             'temp_dir': temp_dir,
#             'cert_folder': cert_folder,
#             'accepted_folder': accepted_folder,
#             'rejected_folder': rejected_folder,
#             'temp_links': temp_links
#         }
        
#     finally:
#         # Cleanup symlinks
#         for link_path in temp_links:
#             if os.path.exists(link_path):
#                 try:
#                     if os.path.islink(link_path):
#                         os.unlink(link_path)
#                     elif os.path.isdir(link_path):
#                         shutil.rmtree(link_path)
#                 except Exception as e:
#                     logger.warning(f"Could not remove {link_path}: {e}")
        
#         # Cleanup temp directory
#         try:
#             shutil.rmtree(temp_dir)
#             logger.info(f"Cleaned up temporary directory: {temp_dir}")
#         except Exception as e:
#             logger.warning(f"Could not clean up temp directory {temp_dir}: {e}")

# def create_in_memory_zip(folder_path: str) -> io.BytesIO:
#     """Create ZIP file in memory from folder contents"""
#     zip_buffer = io.BytesIO()
    
#     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         if os.path.exists(folder_path):
#             files_added = 0
#             for file in os.listdir(folder_path):
#                 file_path = os.path.join(folder_path, file)
#                 if os.path.isfile(file_path):
#                     zipf.write(file_path, file)
#                     files_added += 1
#             logger.info(f"Created in-memory zip with {files_added} files")
    
#     zip_buffer.seek(0)
#     return zip_buffer

# @app.get("/")
# async def root():
#     """Root endpoint with API information"""
#     return {
#         "message": "Certificate Verification API is running on Render!",
#         "version": "2.1",
#         "status": "healthy",
#         "environment": "production",
#         "storage_policy": "Temporary processing only - no files stored permanently",
#         "endpoints": {
#             "upload_and_process": "/upload-and-process",
#             "health_check": "/health",
#             "list_jobs": "/jobs",
#             "job_status": "/status/{job_id}",
#             "download_accepted": "/download/accepted/{job_id}",
#             "download_rejected": "/download/rejected/{job_id}",
#             "api_docs": "/docs"
#         }
#     }

# @app.post("/upload-and-process")
# async def upload_and_process_certificates(files: List[UploadFile] = File(...)):
#     """Upload and process certificate files"""
    
#     if not files or len(files) == 0:
#         raise HTTPException(status_code=400, detail="No files uploaded")
    
#     job_id = str(uuid.uuid4())
#     processing_jobs[job_id] = {
#         "status": "processing", 
#         "message": "Starting certificate processing...",
#         "progress": "Uploading files"
#     }
    
#     try:
#         logger.info(f"Starting job {job_id} with {len(files)} files")
        
#         with temporary_processing_environment() as temp_env:
#             processing_jobs[job_id]["progress"] = "Saving uploaded files to temporary storage"
            
#             saved_files = []
#             allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            
#             for file in files:
#                 if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
#                     raise HTTPException(
#                         status_code=400, 
#                         detail=f"Invalid file type: {file.filename}. Allowed: {', '.join(allowed_extensions)}"
#                     )
                
#                 file_path = os.path.join(temp_env['cert_folder'], file.filename)
#                 with open(file_path, "wb") as buffer:
#                     content = await file.read()
#                     buffer.write(content)
#                 saved_files.append(file.filename)
            
#             logger.info(f"Saved {len(saved_files)} files to temporary storage: {saved_files}")
            
#             processing_jobs[job_id].update({
#                 "message": "Files uploaded to temporary storage, starting processing...",
#                 "progress": "Converting documents to PNG",
#                 "uploaded_files": saved_files
#             })
            
#             logger.info("Starting processing pipeline...")
#             processing_jobs[job_id]["progress"] = "Processing certificates"
            
#             # Run your processing graph
#             final_state = graph.invoke({
#                 "messages": [], 
#                 "ecerti": [], 
#                 "human": [], 
#                 "rejected_certi": [], 
#                 "accepted_certi": [], 
#                 "ocr_texts": {}
#             })
            
#             logger.info("Processing completed")
#             logger.info(f"Results - Accepted: {len(final_state['accepted_certi'])}, Rejected: {len(final_state['rejected_certi'])}")
            
#             processing_jobs[job_id]["progress"] = "Creating result packages in memory"
            
#             accepted_zip_data = None
#             rejected_zip_data = None
            
#             if final_state["accepted_certi"]:
#                 accepted_zip_data = create_in_memory_zip(temp_env['accepted_folder'])
            
#             if final_state["rejected_certi"]:
#                 rejected_zip_data = create_in_memory_zip(temp_env['rejected_folder'])
            
#             total_processed = len(final_state["accepted_certi"]) + len(final_state["rejected_certi"])
            
#             processing_jobs[job_id] = {
#                 "status": "completed",
#                 "message": "Processing completed successfully",
#                 "progress": "Completed",
#                 "uploaded_files": saved_files,
#                 "results": {
#                     "accepted_certificates": final_state["accepted_certi"],
#                     "rejected_certificates": final_state["rejected_certi"],
#                     "e_certificates": final_state["ecerti"],
#                     "human_certificates": final_state["human"],
#                     "total_processed": total_processed,
#                     "accepted_zip_data": accepted_zip_data,
#                     "rejected_zip_data": rejected_zip_data,
#                     "ocr_texts": final_state["ocr_texts"]
#                 }
#             }
        
#         logger.info(f"Temporary processing environment cleaned up for job {job_id}")
        
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
#                 "human_certificates_count": len(final_state["human"])
#             },
#             "download_links": {
#                 "accepted": f"/download/accepted/{job_id}" if accepted_zip_data else None,
#                 "rejected": f"/download/rejected/{job_id}" if rejected_zip_data else None
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
#     """Download accepted certificates as ZIP"""
#     if job_id not in processing_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     job = processing_jobs[job_id]
#     if job["status"] != "completed":
#         raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    
#     zip_data = job["results"].get("accepted_zip_data")
#     if not zip_data:
#         raise HTTPException(status_code=404, detail="No accepted certificates found")
    
#     logger.info(f"Streaming accepted certificates for job {job_id}")
    
#     zip_buffer = io.BytesIO(zip_data.getvalue())
#     zip_buffer.seek(0)
    
#     return StreamingResponse(
#         io.BytesIO(zip_buffer.read()),
#         media_type='application/zip',
#         headers={"Content-Disposition": f"attachment; filename=accepted_certificates_{job_id}.zip"}
#     )

# @app.get("/download/rejected/{job_id}")
# async def download_rejected_certificates(job_id: str):
#     """Download rejected certificates as ZIP"""
#     if job_id not in processing_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     job = processing_jobs[job_id]
#     if job["status"] != "completed":
#         raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    
#     zip_data = job["results"].get("rejected_zip_data")
#     if not zip_data:
#         raise HTTPException(status_code=404, detail="No rejected certificates found")
    
#     logger.info(f"Streaming rejected certificates for job {job_id}")
    
#     zip_buffer = io.BytesIO(zip_data.getvalue())
#     zip_buffer.seek(0)
    
#     return StreamingResponse(
#         io.BytesIO(zip_buffer.read()),
#         media_type='application/zip',
#         headers={"Content-Disposition": f"attachment; filename=rejected_certificates_{job_id}.zip"}
#     )

# @app.get("/status/{job_id}")
# async def get_job_status(job_id: str):
#     """Get job processing status"""
#     if job_id not in processing_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     status = processing_jobs[job_id].copy()
#     if "results" in status and "accepted_zip_data" in status["results"]:
#         status["results"] = {k: v for k, v in status["results"].items() 
#                            if k not in ["accepted_zip_data", "rejected_zip_data"]}
#         status["results"]["download_ready"] = True
    
#     return status

# @app.get("/jobs")
# async def list_all_jobs():
#     """List all processing jobs"""
#     return {
#         "total_jobs": len(processing_jobs),
#         "jobs": {job_id: {
#             "status": job["status"], 
#             "message": job["message"],
#             "has_downloads": job.get("results", {}).get("accepted_zip_data") is not None or 
#                            job.get("results", {}).get("rejected_zip_data") is not None
#         } for job_id, job in processing_jobs.items()}
#     }

# @app.delete("/cleanup/{job_id}")
# async def cleanup_job_files(job_id: str):
#     """Cleanup specific job from memory"""
#     if job_id not in processing_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     del processing_jobs[job_id]
#     logger.info(f"Cleaned up job {job_id} from memory")
    
#     return {"message": "Job cleaned up successfully from memory"}

# @app.delete("/cleanup-all")
# async def cleanup_all_jobs():
#     """Cleanup all jobs from memory"""
#     cleanup_count = len(processing_jobs)
#     processing_jobs.clear()
#     logger.info(f"Cleaned up {cleanup_count} jobs from memory")
    
#     return {"message": f"Cleaned up {cleanup_count} jobs successfully from memory"}

# @app.get("/health")
# async def health_check():
#     """Health check endpoint for Render"""
#     return {
#         "status": "healthy",
#         "version": "2.1",
#         "port": PORT,
#         "active_jobs": len([job for job in processing_jobs.values() if job["status"] == "processing"]),
#         "completed_jobs": len([job for job in processing_jobs.values() if job["status"] == "completed"]),
#         "failed_jobs": len([job for job in processing_jobs.values() if job["status"] == "failed"])
#     }

# # Run the application
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "api:app", 
#         host="0.0.0.0", 
#         port=PORT, 
#         log_level="info",
#         access_log=True,
#         reload=False
#     )




from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
from pathlib import Path
from app import graph, State  # assuming your current code is in app.py
from PIL import Image
import io

app = FastAPI(title="Certificate Verification API")

# Define folder structure
BASE_DIR = Path("./")
UPLOAD_DIR = BASE_DIR / "uploaded_certificates"
PROCESSED_DIR = BASE_DIR / "processed_certificates"
ACCEPTED_DIR = BASE_DIR / "accepted_certificates"
REJECTED_DIR = BASE_DIR / "rejected_certificates"

# Ensure directories exist
for folder in [UPLOAD_DIR, PROCESSED_DIR, ACCEPTED_DIR, REJECTED_DIR]:
    folder.mkdir(exist_ok=True)

@app.post("/upload-certificates")
async def upload_certificates(files: List[UploadFile] = File(...)):
    """
    Upload multiple certificate files.
    """
    uploaded_files = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
        uploaded_files.append(file.filename)
    return {"uploaded_files": uploaded_files}


@app.post("/process-certificates")
def process_certificates():
    """
    Run the certificate verification pipeline on uploaded files.
    """
    try:
        # Clear processed folder
        if PROCESSED_DIR.exists():
            shutil.rmtree(PROCESSED_DIR)
        PROCESSED_DIR.mkdir()

        # Copy uploaded files to processed folder
        for file in UPLOAD_DIR.iterdir():
            if file.is_file():
                shutil.copy(file, PROCESSED_DIR / file.name)

        # Initialize state
        initial_state = {
            "messages": [],
            "ecerti": [],
            "human": [],
            "rejected_certi": [],
            "accepted_certi": [],
            "ocr_texts": {}
        }

        # Run the LangGraph pipeline
        final_state = graph.invoke(initial_state)

        # Save accepted and rejected certificates
        for certi in final_state["accepted_certi"]:
            src = PROCESSED_DIR / certi
            dst = ACCEPTED_DIR / certi
            if src.exists():
                with Image.open(src) as img:
                    img.save(dst, "PNG")

        for certi in final_state["rejected_certi"]:
            src = PROCESSED_DIR / certi
            dst = REJECTED_DIR / certi
            if src.exists():
                with Image.open(src) as img:
                    img.save(dst, "PNG")

        return JSONResponse(content={
            "accepted_certificates": final_state["accepted_certi"],
            "rejected_certificates": final_state["rejected_certi"],
            "messages": final_state["messages"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-certificates")
def list_certificates():
    """
    List all uploaded, accepted, and rejected certificates.
    """
    uploaded = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    accepted = [f.name for f in ACCEPTED_DIR.iterdir() if f.is_file()]
    rejected = [f.name for f in REJECTED_DIR.iterdir() if f.is_file()]

    return {
        "uploaded": uploaded,
        "accepted": accepted,
        "rejected": rejected
    }
