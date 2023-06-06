import os
import get_folder_and_file as get
path = r'C:\Users\guote\PycharmProjects\sampleselection\Optical Flow\backup\Optical Flow without masks\Optical Flow Frame for Images\c'
path2 = os.path.join(path,'CASME2')
path3 = os.path.join(path,'CASMEII')
path4 = r'C:\Users\guote\PycharmProjects\sampleselection\Optical Flow\backup\maskedCASMEII\cropped'
folders = get.get_folders(path4)

print(folders)
