import os

# check the length of the output file
def check_length():
    output_path = "output_img"
    output_files = os.listdir(output_path)
    print(len(output_files)/2)

if __name__ == "__main__":
    check_length()