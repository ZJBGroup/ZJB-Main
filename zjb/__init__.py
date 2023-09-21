from pkgutil import extend_path

try:
    __path__ = extend_path(__path__, __name__)

except NameError:
    # Ignore __path__ not defined when called from sphinx
    __path__ = [__name__]
