PROJECT = izvjestaj
IMAGES = $(wildcard images/*.pdf)
CITATIONS = literatura.bib

all: $(PROJECT).pdf

$(PROJECT).pdf: $(PROJECT).tex $(IMAGES) $(CITATIONS)
	latexmk -pdf -pdflatex="pdflatex -halt-on-error"  $(PROJECT)

cleanall:
	latexmk -C

clean:
	latexmk -c
