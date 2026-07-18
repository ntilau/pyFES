TEX = paper.tex
PDF = $(TEX:.tex=.pdf)
AUX = $(TEX:.tex=.aux)

LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all pdf view clean cleanall

all: pdf

pdf: $(PDF)

$(PDF): $(TEX)
	$(LATEX) $<
	-@grep -q "^\\citation{" $(AUX) 2>/dev/null && bibtex $(AUX:.aux=) 2>/dev/null || true
	$(LATEX) $<
	@while grep -q "Rerun to get cross-references right" $*.log 2>/dev/null; do \
		echo "  Rerunning LaTeX (cross-references)..."; \
		$(LATEX) $<; \
	done
	@echo ""
	@echo "── Paper compiled: $(PDF) ──"
	@echo ""

view: $(PDF)
	open $(PDF)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
	rm -f *.dvi *.ps *.synctex.gz *.run.xml *-blx.bib *.bcf

cleanall: clean
	rm -f $(PDF)
