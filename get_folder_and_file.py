import os
import fnmatch


def get_filelist(dir, Filelist):
	newDir = dir
	if os.path.isfile(dir):
		Filelist.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			newDir = os.path.join(dir, s)
			get_filelist(newDir, Filelist)
	filenum = len(Filelist)
	return Filelist


def file_ext(filename, level=1):
	"""
	return extension of filename

	Parameters:
	-----------
	filename: str
		name of file, path can be included
	level: int
		level of extension.
		for example, if filename is 'sky.png.bak', the 1st level extension
		is 'bak', and the 2nd level extension is 'png'

	Returns:
	--------
	extension of filename
	"""
	return filename.split('.')[-level]


def _contain_file(path, extensions):
	"""
	check whether path contains any file whose extension is in extensions list

	Parameters:
	-----------
	path: str
		path to be checked
	extensions: str or list/tuple of str
		extension or extensions list

	Returns:
	--------
	return True if contains, else return False
	"""
	assert os.path.exists(path), 'path must exist'
	assert os.path.isdir(path), 'path must be dir'
	
	if isinstance(extensions, str):
		extensions = [extensions]
	
	for file in os.listdir(path):
		if os.path.isfile(os.path.join(path, file)):
			if (extensions is None) or (file_ext(file) in extensions):
				return True
	return False


def _process_extensions(extensions=None):
	"""
	preprocess and check extensions, if extensions is str, convert it to list.

	Parameters:
	-----------
	extensions: str or list/tuple of str
		file extensions

	Returns:
	--------
	extensions: list/tuple of str
		file extensions
	"""
	if extensions is not None:
		if isinstance(extensions, str):
			extensions = [extensions]
		assert isinstance(extensions, (list, tuple)), \
			'extensions must be str or list/tuple of str'
		for ext in extensions:
			assert isinstance(ext, str), 'extension must be str'
	return extensions


def get_folders(path,  extensions=None, is_recursive=True):
	"""
	read folders in path. if extensions is None, read all folders, if
	extensions are specified, only read the folders who contain any files that
	have one of the extensions. if is_recursive is True, recursively read all
	folders, if is_recursive is False, only read folders in current path.

	Parameters:
	-----------
	path: str
		path to be read
	extensions: str or list/tuple of str
		file extensions
	is_recursive: bool
		whether read folders recursively. read recursively is True, while just
		read folders in current path if False

	Returns:
	--------
	folders: the obtained folders in path
	"""
	extensions = _process_extensions(extensions)
	folders = []
	# get folders in current path
	if not is_recursive:
		for name in os.listdir(path):
			fullname = os.path.join(path, name)
			print(fullname)
			if os.path.isdir(fullname):
				if (extensions is None) or (_contain_file(fullname, extensions)):
					folders.append(fullname)
		return folders
	
	# get folders recursively
	for main_dir, _, _ in os.walk(path):
		if (extensions is None) or (_contain_file(main_dir, extensions)):
			folders.append(main_dir)

	"""确定最长的目录有几级，然后去掉所有父目录"""
	level = 1
	levels = []
	final = []
	for folder in folders:
		level = 1
		for i in range(len(folder)):
			if folder[i] == '\\':
				level += 1
			else:
				continue
		levels.append(level)
	
	for folder in folders:
		level = 1
		for i in range(len(folder)):
			if folder[i] == '\\':
				level += 1
			else:
				continue
		if level < max(levels):
			continue
		elif max(levels) == 1:
			final.append(folder)
		else:
			final.append(folder)
			
	return final