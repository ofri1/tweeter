from tkinter import *
import re
import urllib.request
import os, shutil


class Downloader(Frame):

    def download(self):

        SPECIES_LIST = []
        NUM_PAGES_LIST = []

        for i in range(self.n):
            SPECIES_LIST.append(self.entry[2*i].get())
            NUM_PAGES_LIST.append(int(self.entry[2 * i+1].get()))

        for species, num_pages in zip(SPECIES_LIST, NUM_PAGES_LIST):

            if not os.path.exists('files/' + species):
                os.makedirs('files/' + species)
                os.makedirs('files/' + species + '/src')

            i = 1  # Records counter, for naming files

            for j in range(num_pages):
                if not os.path.isfile('files/' + species + '/src/page_' + str(j + 1)):  # Download HTML source
                    urllib.request.urlretrieve(
                        'https://www.xeno-canto.org/species/' + species + '?view=0&pg=' + str(j + 1),
                        'files/' + species + '/src/page_' + str(j + 1))
                with open('files/' + species + '/src/page_' + str(j + 1), 'r', encoding="utf8") as file:
                    text = file.read()
                    files = re.findall('//www.*\.mp3', text)  # Find all mp3 links
                for url in files:
                    if not os.path.isfile('files/' + species + '/rec_' + str(i) + '.mp3'):  # Download mp3
                        urllib.request.urlretrieve('http:' + url, 'files/' + species + '/rec_' + str(i) + '.mp3')
                    i += 1

            shutil.rmtree('files/' + species + '/src')
            print("Successfully downloaded " + str(i) + " files for species " + species)
            self.quit()

    def createWidgets(self):

        self.numentry = Entry(self)
        self.numentry.grid(row=0,column=0,sticky=W)

        self.GO = Button(self)
        self.GO["text"] = "ADD"
        self.GO["command"] = self.add_entries
        self.GO.grid(row=0,column=1,sticky=W)

    def add_entries(self):

        self.n = int(self.numentry.get())

        self.numentry.destroy()
        self.GO.destroy()

        self.entry = []
        for i in range(self.n):
            self.entry.append(Entry(self))
            self.entry.append(Entry(self))
            self.entry[2*i].grid(row=i+1,column=0,sticky=W)
            self.entry[2*i+1].grid(row=i + 1, column=1, sticky=W)

        self.DL = Button(self,text="DOWNLOAD",command=self.download)
        self.DL.grid(row=self.n+1,column=0,sticky=W)


    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

root = Tk()
app = Downloader(master=root)
app.mainloop()
root.destroy()