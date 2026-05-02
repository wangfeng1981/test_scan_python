def myfunc(fname):
    with open(fname, "r") as file:
        first_10_chars = file.read(10) # Reads first 10 characters
        print(first_10_chars)

if __name__=="__main__":
	myfunc("my.py")