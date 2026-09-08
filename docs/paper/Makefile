all: qc.pdf

qc.pdf: main.tex latexmkrc
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C

.PHONY: all clean
