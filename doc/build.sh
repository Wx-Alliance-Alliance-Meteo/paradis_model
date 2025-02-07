#!/bin/bash
filename=main
pdflatex $filename && \
bibtex $filename   && \
pdflatex $filename && \
pdflatex $filename
