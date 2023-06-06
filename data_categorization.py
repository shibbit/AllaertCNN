import shutil

from get_folder_and_file import get_folders, get_filelist
import os
import xlsread
import fnmatch
import re
path = r'C:\Users\guote\PycharmProjects\unetAE\resultsnotuning'
output = r'./AEnotuning'
pattern1 = "sub*"
pattern2 = "*EP*"


def categorize(sourcepath, resultpath):  # CASME 2
	annot = xlsread.get_all_annotations(r'./CASME2.xls')
	folders = get_folders(sourcepath)
	Filename = []
	if len(folders) == 1:
		list = os.listdir(sourcepath)
		if fnmatch.fnmatch(list[0], pattern1) and fnmatch.fnmatch(list[0], pattern2):
			for index, value in enumerate(list):
				subid = value[3:5]
				sub = value[0:5]
				for i in range(len(value)):
					if value[i:i+4] == '-reg':
						Filename = value[6:i]
				for entries in annot:
					if entries.get('Subject') == subid and entries.get('Filename') == Filename:
						emotion = entries.get('Estimated Emotion')
						category = os.path.normpath(os.path.join(os.getcwd(), resultpath, emotion))
						try:
							os.makedirs(category, exist_ok=True)
						except OSError as error:
							print("Directory '%s' can not be created")
						shutil.move(os.path.abspath(os.path.normpath(os.path.join(sourcepath, value))), os.path.join(category))
	else:
		for folder in folders:
			parenfolder = os.path.dirname(os.path.dirname(folder))
			subid = folder.replace(parenfolder + '\\', '')[3:5]
			sub = folder.replace(parenfolder + '\\', '')[0:5]
			Filename = folder.replace(parenfolder, '').replace('\\' + sub + '\\', '')
			for entries in annot:
				if entries.get('Subject') == subid and entries.get('Filename') == Filename:
					files = get_filelist(folder, [])
					emotion = entries.get('Estimated Emotion')
					category = os.path.normpath(os.path.join(os.getcwd(), resultpath, emotion))
					try:
						os.makedirs(category, exist_ok=True)
					except OSError as error:
						print("Directory '%s' can not be created")
					for file in files:  # ./Optical Flow/CASME2\Cropped\sub26\EP18_49\reg_img16.jpg
						eachfilename = file.replace(folder + '\\', '').replace('.jpg', '')
						newname = sub + '-' + Filename + '-' + eachfilename + '-' + emotion + '.jpg'
						print("Processing:", file)
						os.rename(os.path.abspath(os.path.normpath(file)), os.path.join(category + '\\' + newname))


categorize(path, output)
