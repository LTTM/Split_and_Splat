import os
import shutil
import argparse

# Path to the "ref" folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a ScanNet scene")
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name, e.g. scene0200_00"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output folder--, e.g. scene0200_00"
    )
    
    
    parser.add_argument("--init", action="store_true")
     
    args = parser.parse_args()

    SCENE = args.scene
    
    IS_INIT = args.init
    
    OUTPUT = args.output

    #========== MOVE PLY ==========
    
    if(IS_INIT): path = "ref"
    else: path ="comb"
    raw_iter = 1000
    ref_path = "./output/"+SCENE+"/"+ path
    comb_iter = 1000
    ""
    output_folder = "./output/"+SCENE+"/" + OUTPUT
    
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    for parent in os.listdir(ref_path):
        parent_path = os.path.join(ref_path, parent)
        if not os.path.isdir(parent_path):
            
            continue  # skip non-folders

        # path to the expected PLY file
        if path!="raw":
            ply_path = os.path.join(parent_path, "point_cloud", f"iteration_{comb_iter}", "point_cloud.ply")
        else:
            ply_path = os.path.join(parent_path, "point_cloud", f"iteration_{raw_iter}", "point_cloud.ply")
            print(ply_path)

            

        if os.path.isfile(ply_path):
            # new filename: parent folder name + .ply
            new_name = f"{parent}.ply"
            dest_path = os.path.join(output_folder, new_name)

            check_name = os.path.join(output_folder,"old", new_name)
            if os.path.exists(dest_path) or os.path.isfile(check_name):
                continue
            # Move the PLY file

            shutil.copy(ply_path, dest_path)
            print(f"Moved: {ply_path} → {dest_path}")
        else:
            print(f"PLY not found for: {parent}")
            
     #========== MOVE MASKS ==========
     
    ref_path = "./"+SCENE+"/masks/combined"
    os.makedirs(ref_path, exist_ok=True)
    output_folder = "./"+SCENE+"/masks"
    
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    for parent in os.listdir(ref_path):
        mask_path = os.path.join(ref_path, parent)
        if not os.path.isdir(mask_path):
            continue  # skip non-folders

        # path to the expected PLY file
        

        if os.path.isdir(mask_path):
            # new filename: parent folder name + .ply
            dest_path = os.path.join(output_folder, parent)

            check_name = os.path.join(output_folder, parent)
            if os.path.exists(dest_path) or os.path.isfile(check_name):
                continue
            # Move the PLY file

            shutil.move(mask_path, dest_path)
            print(f"Moved: {mask_path} → {dest_path}")
        else:
            print(f"MASK not found for: {parent}")

