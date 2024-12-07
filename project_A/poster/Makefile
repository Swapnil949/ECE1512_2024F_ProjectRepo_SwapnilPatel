.PHONY: main clean FORCE

main: ECE1512_Project_A_poster.pdf

ECE1512_Project_A_poster.pdf: FORCE
	latexmk -pdflatex='lualatex -interaction nonstopmode' -pdf ECE1512_Project_A_poster.tex

clean:
	latexmk -pdf -C
