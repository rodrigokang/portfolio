# Import libraries
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import PyPDF2
import os

# Define paths
cover_image = "cover.png"
main_pdf = "docs/Data-Science-Across-Domains.pdf"
cover_pdf = "cover.pdf"
output_pdf = "data-science-across-domains.pdf"

# Create cover PDF from image
c = canvas.Canvas(cover_pdf, pagesize=A4)
width, height = A4

# Load and draw cover image to fill entire A4 page
try:
    img = ImageReader(cover_image)
    c.drawImage(img, 0, 0, width=width, height=height)
except FileNotFoundError:
    print(f"Error: Cover image '{cover_image}' not found.")
    exit(1)

# Save cover PDF
c.save()

# Function to merge PDFs
def merge_pdfs(cover_path, content_path, output_path):
    # Check if files exist
    if not os.path.exists(content_path):
        print(f"Error: Main PDF '{content_path}' not found.")
        return False
    
    merger = PyPDF2.PdfMerger()
    
    try:
        # Add cover
        merger.append(cover_path)
        
        # Add main content
        merger.append(content_path)
        
        # Save unified PDF
        merger.write(output_path)
        merger.close()
        print(f"Successfully created '{output_path}'")
        return True
        
    except Exception as e:
        print(f"Error merging PDFs: {e}")
        merger.close()
        return False

# Merge the PDFs
merge_pdfs(cover_pdf, main_pdf, output_pdf)