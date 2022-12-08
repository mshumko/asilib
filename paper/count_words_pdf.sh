#!/bin/bash

pdftotext "$1" - | sed -n "/Introduction/,/References/p" | wc -w
