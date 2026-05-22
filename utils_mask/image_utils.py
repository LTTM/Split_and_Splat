import os
from PIL import Image



def JPEGtoPng(input_folder, output_folder):
    """
    Convert all JPEG images in the input folder to PNG format and save them in the output folder.
    
    Parameters:
    input_folder (str): Path to the folder containing JPEG images.
    output_folder (str): Path to the folder where PNG images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(input_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpeg'):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(output_folder, png_filename)
            img.save(png_path)
            print(f"Saved {png_path}")


def set_white_background(input_path, output_path):
    """Composite an RGBA image onto a white background and save as RGB PNG."""
    # Open the image
    img = Image.open(input_path).convert("RGBA")
    # Create a white background image
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    # Composite the original image onto the white background
    combined = Image.alpha_composite(white_bg, img)
    # Save the result as a PNG
    combined.convert("RGB").save(output_path, "PNG")





