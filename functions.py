import os
import shutil
#import win32com.client
import fpdf
from fpdf import FPDF
#import pythoncom
from PyPDF2 import PdfMerger



from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
#from wandb import config

def delete_all(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)  # Delete the file

    print("All files have been deleted.")

def save_uploaded_file(uploaded_files):
    # Define the path where to save the file on the server
    if not os.path.exists("uploaded_files"):
        os.makedirs("uploaded_files")

    for file in uploaded_files:
        save_path = os.path.join("uploaded_files", file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

def delete_and_save(uploaded_files, path):
    delete_all(path)
    save_uploaded_file(uploaded_files)


def get_config():
    return {
            "upload_path": "uploaded_files",
            "processed_path" : "processed_files",
            "merge_pdf" :  "processed_files/merged_output.pdf"

        }

def convert_docx_to_pdf(docx_path, pdf_path):
    pythoncom.CoInitialize()
    word = win32com.client.Dispatch("Word.Application")
    doc = word.Documents.Open(docx_path)
    doc.SaveAs(pdf_path, FileFormat=17)  # 17 is the format for PDFs
    doc.Close()
    word.Quit()
    print(f"Converted {docx_path} to {pdf_path}")


def txt_to_pdf(txt_file_path, pdf_file_path):
    c = canvas.Canvas(pdf_file_path, pagesize=letter)

    c.setFont("Helvetica", 12)

    text_object = c.beginText(40, 750)  # Starting from the top-left corner
    text_object.setFont("Helvetica", 12)

    with open(txt_file_path, "r", encoding="utf-8") as file:
        for line in file:
            text_object.textLine(line.strip())  # Add each line to the PDF

    c.drawText(text_object)

    # Save the PDF file
    c.save()
    print(f"PDF created successfully: {pdf_file_path}")

def convert_to_pdf(upload_path, processed_path):
    for file in os.listdir(upload_path):
        file_path = os.path.join(upload_path, file)
        _, extension = os.path.splitext(file)
        if not os.path.exists("processed_files"):
            os.makedirs("processed_files")
        new_file_path = os.path.join("processed_files", file)


        if extension.lower() in ['.pdf']:

            shutil.copy(file_path, new_file_path)

        elif extension.lower() in ['.docx']:
        #file_path = file_path.replace('\\', '\\\\')
        #new_file_path =  new_file_path.replace('\\', '\\\\')
            docx_file = upload_path + "\\" + file
            pdf_file = processed_path + "\\" + _ + ".pdf"
            convert_docx_to_pdf(docx_file, pdf_file)

        elif extension.lower() in ['.txt']:
            pdf_file = processed_path + "\\" + _ + ".pdf"
            txt_to_pdf(file_path, pdf_file)


def merge_pdf(merge_path, output_file):
    #merger.open()
# Iterate over all files in the folder
    with PdfMerger() as merger:
        for filename in os.listdir(merge_path):
            if filename.endswith(".pdf"):  # Check for PDF files
                file_path = os.path.join(merge_path, filename)
                merger.append(file_path)

        merger.write(output_file)
        #merger.close()

    print(f"All PDFs in the folder merged successfully into {output_file}")