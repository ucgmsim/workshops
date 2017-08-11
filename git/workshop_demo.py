import glob
import os

def print_names():
	print "The people who have done the Git workshop are:"
	with open('names') as fp:
		for line in fp:
			print line
        files = glob.glob("names_dir/*")
	print "People who are in progress in the workshop are:"
        for file in files:
            print os.path.basename(file)

if __name__ == '__main__':
	print_names()
