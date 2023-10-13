import typer

from .generation import generation


app = typer.Typer()


@app.command()
def __version():
    import platform
    import sys

    print(f"Python version:\t\t{sys.version.split()[0]}")

    print(f"OS/Arch:\t\t{platform.system().lower()}/{platform.machine().lower()}")


app.command()(generation)


if __name__ == "__main__":
    app()
