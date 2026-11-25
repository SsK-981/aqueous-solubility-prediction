"""Map source CSV schemas to the cleaner's input fields."""

BIGSOL_MARKERS = {"SMILES_Solute", "Temperature_K"}
SOMAS_MARKERS = {"SMILES", "Temperature", "Experimental Solubility in Water"}
AQ_MARKERS = {"SmilesCurated", "ExperimentalLogS"}
DLS_MARKERS = {"SMILES", "LogS exp (mol/L)"}
SOURCES = ("auto", "bigsoldb", "somas", "aqsoldbc", "dls100", "mapped")


def detect_source(headers):
    columns = set(headers)
    if BIGSOL_MARKERS <= columns and {"Solvent", "SMILES_Solvent"} & columns:
        return "bigsoldb"
    if SOMAS_MARKERS <= columns:
        return "somas"
    if AQ_MARKERS <= columns:
        return "aqsoldbc"
    if DLS_MARKERS <= columns:
        return "dls100"
    if {"smiles", "temperature", "temperature_unit"} <= columns:
        return "standard"
    raise ValueError("CSV format not recognized. Select --source or use --source mapped with column options.")


def _water_filter(columns):
    def check(row):
        labels = [row[column].strip().casefold() for column in columns if row[column].strip()]
        if not labels:
            return "solvent is missing"
        known_water = {"water", "h2o", "o", "[oh2]"}
        is_water = all(label in known_water for label in labels)
        return "" if is_water else "solvent is outside the water-only prediction scope"

    return check


def _mapped_fields(args):
    temperature_column = args.get("temperature_column")
    temperature_unit_column = args.get("temperature_unit_column")
    temperature_unit = args.get("temperature_unit")
    measured_column = args.get("solubility_column")
    measured_unit_column = args.get("solubility_unit_column")
    measured_unit = args.get("solubility_unit")
    if not args.get("smiles_column"):
        raise ValueError("--source mapped requires --smiles-column")
    if bool(temperature_column) != bool(temperature_unit or temperature_unit_column):
        raise ValueError("Map the temperature column and provide its units together")
    if bool(measured_column) != bool(measured_unit or measured_unit_column):
        raise ValueError("Map the solubility column and provide its units together")
    return (
        {"smiles": args["smiles_column"], "temperature": temperature_column,
         "temperature_unit": temperature_unit_column, "solubility": measured_column,
         "solubility_unit": measured_unit_column},
        {"temperature": "", "temperature_unit": temperature_unit or "",
         "solubility_unit": measured_unit or ""},
    )


def normalize(headers, rows, *, source="auto", use_release_molarity=False, **mapping):
    """Return the schema name, preserved source fields, normalized rows and water check."""
    columns = set(headers)
    if source == "auto":
        source = detect_source(headers)
    if any(mapping.values()) and source != "mapped":
        raise ValueError("Column mapping options require --source mapped")
    if use_release_molarity and source != "bigsoldb":
        raise ValueError("--use-release-molarity applies only to BigSolDB")

    field_map = {}
    constants = {}
    water_columns = []
    if source == "bigsoldb":
        water_columns = [column for column in ("Solvent", "SMILES_Solvent") if column in columns]
        if not water_columns:
            raise ValueError("BigSolDB input needs Solvent or SMILES_Solvent")
        field_map = {"smiles": "SMILES_Solute", "temperature": "Temperature_K"}
        constants = {"temperature_unit": "K"}
        if use_release_molarity:
            field_map["solubility"] = "Solubility(mol/L)"
            constants["solubility_unit"] = "mol/L"
            constants["solubility_basis"] = "BigSolDB release molarity (derived concentration)"
        elif "Solubility(mole_fraction)" in columns:
            field_map["solubility"] = "Solubility(mole_fraction)"
            constants["solubility_unit"] = "mole fraction"
    elif source == "somas":
        field_map = {"smiles": "SMILES", "temperature": "Temperature",
                     "solubility": "Experimental Solubility in Water"}
        constants = {"temperature_unit": "K", "solubility_unit": "mg/L"}
        if "Molar Mass" in columns:
            field_map["solubility_molar_mass_g_mol"] = "Molar Mass"
    elif source in {"aqsoldbc", "dls100"}:
        if source == "aqsoldbc":
            field_map = {"smiles": "SmilesCurated", "solubility": "ExperimentalLogS"}
        else:
            field_map = {"smiles": "SMILES", "solubility": "LogS exp (mol/L)"}
        constants = {"temperature": "", "temperature_unit": "", "solubility_unit": "log10_mol/L"}
    elif source == "mapped":
        field_map, constants = _mapped_fields(mapping)
        solvent_column = mapping.get("solvent_column")
        if solvent_column:
            water_columns = [solvent_column]
    elif source != "standard":
        raise ValueError(f"Unsupported dataset source: {source}")

    field_map = {field: column for field, column in field_map.items() if column is not None}
    constants = {field: value for field, value in constants.items()
                 if field not in field_map and (field != "solubility_unit" or "solubility" in field_map)}
    missing = set(field_map.values()).union(water_columns) - columns
    if missing:
        raise ValueError("Missing source columns: " + ", ".join(sorted(missing)))
    collisions = ((set(field_map) | set(constants)) & columns) - {
        field for field, column in field_map.items() if field == column
    }
    if collisions:
        raise ValueError("Source mapping would replace existing columns: " + ", ".join(sorted(collisions)))
    added = list(dict.fromkeys([*field_map, *constants]))
    output_headers = [*headers, *(field for field in added if field not in columns)]
    normalized = [
        {**row, **constants, **{field: row[column] for field, column in field_map.items()}}
        for row in rows
    ] if added else rows
    water_check = _water_filter(water_columns) if water_columns else None
    return source, output_headers, normalized, water_check
