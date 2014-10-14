__author__ = 'samminns'
import os, fnmatch, glob

output = {}
data_dir = os.getcwd()
print data_dir
file_names = ['00', '15', '30', '60']
type_name = ['test', 'train']


def convert(x):
    out = []
    for i in range(10):
        out.append('0.0')
    out[int(x[0])] = '1.0'
    return out


def convert_file():
    for h in range(4):
        for g in range(2):
            o = open("digits_" + file_names[h] + "_" + type_name[g] + "_bpnn.pat", 'w')
            f = open("digits_" + file_names[h] + "_" + type_name[g], 'r+')
            for line in f.readlines():
                chars = line.split(',')
                end = len(chars)
                out_str = ""
                for c in range(end):
                    if c < end - 1:
                        out_str += chars[c] + ".0 "
                    else:
                        s = chars[c].split('\n')
                        bin_str = convert(s)
                        for j in bin_str:
                            out_str += j + " "

                out_str += '\n'
                o.write(out_str)


def main():
    convert_file()

if __name__ == '__main__':
    main()


