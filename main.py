from image_stitching import panoramic_gui, load_images, stitch_images, save_image
import argparse

def run_cli():
    """Command-line interface for the panoramic stitching tool."""
    # Ask the user to input the file paths
    image_input = input("Please enter the paths to the images separated by spaces: ")

    # Initialize argparse with the user-provided input
    parser = argparse.ArgumentParser(description="Panoramic Image Stitching Tool")
    parser.add_argument("images", nargs="+", help="Paths to input images for stitching")
    parser.add_argument("--output", default="panorama_output.jpg", help="Path to save the stitched panorama")

    # Simulate the command line input using sys.argv
    import sys
    sys.argv = [''] + image_input.split() + sys.argv[1:]  # Adjust the system arguments to include the user input

    args = parser.parse_args()

    try:
        # Load images
        print("Loading images...")
        images = load_images(args.images)
        
        # Stitch images
        print("Stitching images...")
        stitched_image = stitch_images(images)
        
        # Save the result
        save_image(stitched_image, args.output)
        print(f"Panorama saved at {args.output}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function to run the tool."""
    print("Welcome to the Panoramic Image Stitching Tool!")
    print("Choose a mode to run the tool:")
    print("1: Graphical User Interface (GUI)")
    print("2: Command-Line Interface (CLI)")
    
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        panoramic_gui()  # Launch the GUI
    elif choice == "2":
        run_cli()  # Launch the CLI
    else:
        print("Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()
