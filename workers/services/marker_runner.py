"""Run Marker on one PDF, in a child process started by PDFToMarkdownService.

Started by file path (not `-m workers...`) so it imports nothing from the worker. Its config, API keys
included, arrives as JSON in the MARKER_RUNNER_CONFIG environment variable, never on the command line
(where other processes could read it). Does exactly what Marker's own `marker_single` CLI does.
"""
import json
import os
import sys


def main() -> int:
    # pop: Marker's own child processes (its local inference server) don't inherit the keys
    config = json.loads(os.environ.pop("MARKER_RUNNER_CONFIG"))
    pdf_path = config.pop("pdf_path")

    from marker.config.parser import ConfigParser
    from marker.models import create_model_dict
    from marker.output import save_output

    parser = ConfigParser(config)
    converter_cls = parser.get_converter_cls()
    converter = converter_cls(
        config=parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=parser.get_processors(),
        renderer=parser.get_renderer(),
        llm_service=parser.get_llm_service(),
    )
    rendered = converter(pdf_path)
    save_output(rendered, parser.get_output_folder(pdf_path), parser.get_base_filename(pdf_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
