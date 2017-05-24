def print_names():
	print "The people who have done the Git workshop are:"
	with open('names') as fp:
		for line in fp:
			print line 

if __name__ == '__main__':
	print_names()
