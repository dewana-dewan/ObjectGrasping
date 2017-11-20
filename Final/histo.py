def build_histogram(img_arr, bin_size):
	bins_arr = []
	bc = []

	for i in range(int(255/bin_size) + 1):
		#print(i)
		bins_arr.append(0)
		# bc.append(i * bin_size)

	# print(bins_arr)
	#print(img_arr.max(), img_arr.min())

	for i in range(img_arr.shape[0]):
		for j in range(img_arr.shape[1]):
			# print (img_arr[i][j])
			a = int(img_arr[i][j] / bin_size)
			b = img_arr[i][j] % bin_size
			k = abs(0.5 - b / bin_size)
			# print(a, b, img_arr[i][j])

			if (b <= bin_size / 2):
				if (a == 0):
					bins_arr[0] += 1
				else:
					bins_arr[a] += 1 - k
					bins_arr[a - 1] += k
			else:
				bins_arr[a] += 1 - k
				bins_arr[a + 1] += k

	c = 0
	for i in range(int(255/bin_size) + 1):
		c += bins_arr[i]
	# for i in range(int(255/bin_size) + 1):
	# 	bins_arr[i] /= c

	#print(bins_arr, c)
	return bins_arr
