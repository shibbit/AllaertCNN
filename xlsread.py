import xlrd


def get_all_filenames(xlsfile,filelist):
	workbook = xlrd.open_workbook(xlsfile)
	ws = workbook.sheet_by_index(0)
	nrows = ws.nrows
	col_length = nrows
	for i in range(col_length):
		ii=i+1
		if (ii<col_length):
			filenames =ws.row_values(rowx=ii,start_colx=0,end_colx=1)
			filelist.append(filenames)
			# filelist[ii] = ','.join(filelist[ii])
		else:
			break
	return filelist


def get_all_annotations(file,sheet_index=0,ws_header_row=0):
	workbook = xlrd.open_workbook(file)
	ws = workbook.sheet_by_index(0)
	nrows = ws.nrows
	# 获取表头行的信息，为一个列表
	header_row_data = ws.row_values(ws_header_row)
	# 将每行的信息放入一个字典，再将字典放入一个列表中
	list = []
	for rownum in range(1, nrows):
		rowdata = ws.row_values(rownum)
		# 如果rowdata有值，
		if rowdata:
			dict = {}
			for j in range(0, len(header_row_data)):
				# 将excel中的数据分别设置成键值对的形式，放入字典；
				dict[header_row_data[j]] = rowdata[j]
			list.append(dict)
	return list


if __name__ == '__main__':
	Filelist = get_all_filenames([])
	test = get_all_annotations()
	for listElement in test:
		print('%s 的类型是：%s' % (listElement, type(listElement)))



