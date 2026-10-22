"""Model backend entry point. Defaults to printing a plan without opening the model."""
from backends.prism import main

if __name__ == '__main__':
    raise SystemExit(main())
