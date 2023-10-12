import typer

from .process import process


app = typer.Typer()


@app.command()
def version():
    import platform
    import sys

    print(f"Python version:\t\t{sys.version.split()[0]}")

    print(f"OS/Arch:\t\t{platform.system().lower()}/{platform.machine().lower()}")


app.command()(process)


if __name__ == "__main__":
    app()
