# Releasing

Maintainer notes. A release is a tag plus a GitHub release; publishing to PyPI is a
separate, deliberate step.

## Before tagging

```bash
pytest -m "not integration"                 # the full suite, green
python tools/make_figures.py --check        # figures still match their evidence
python -m build                             # wheel and sdist
```

Then install the wheel into a clean virtual environment **outside the checkout** and run
the quick start, so the test cannot pass by importing the source tree:

```bash
python -m venv /tmp/ironmule-check
/tmp/ironmule-check/bin/pip install dist/ironmule-<version>-py3-none-any.whl
cd /tmp && /tmp/ironmule-check/bin/ironmule doctor
/tmp/ironmule-check/bin/ironmule models list
/tmp/ironmule-check/bin/ironmule serve --help
```

CI does the same on 3.11 and 3.12 for every pull request. A red main is fixed before
anything is tagged.

## Tagging

`ironmule/_version.py` is the single source of truth for the version. The tag is
`v<version>`, annotated, on a commit whose CI is green:

```bash
git tag -a v0.1.0 -m "IronMule 0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "IronMule 0.1.0" --notes-file <notes>
```

The release notes embed the headline figure. Render it from the committed SVG rather
than attaching a screenshot:

```bash
rsvg-convert -w 1720 docs/assets/headline-ratios.svg -o headline-ratios.png
gh release upload v0.1.0 headline-ratios.png
```

## Publishing to PyPI

Not automated, and not done from a machine that holds research data by accident:

```bash
python -m twine check dist/*
python -m twine upload dist/*
```

The project is not on PyPI yet. Until it is, the README install path is the checkout.

## The social preview

GitHub takes the social preview image from repository settings, not from the tree.
Upload `docs/assets/social-preview.png` under Settings, General, Social preview. The
SVG beside it is the source; regenerate the PNG with
`rsvg-convert -w 1280 -h 640 docs/assets/social-preview.svg -o docs/assets/social-preview.png`.
