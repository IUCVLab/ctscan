from pathlib import Path

resource_folder = Path("data")

all = (
    resource_folder / "01_Samodurov/DICOM/PA000000/ST000000/SE000003",
    resource_folder / "01_Samodurov/DICOM/PA000001/ST000000/SE000003",
    resource_folder / "01_Samodurov/DICOM/PA000001/ST000000/SE000004",
    resource_folder / "01_Samodurov/DICOM/PA000001/ST000000/SE000006",
    resource_folder / "01_Samodurov/DICOM/PA000001/ST000000/SE000007",
    resource_folder / "01_Samodurov/DICOM/PA000001/ST000000/SE000008",
    resource_folder / "01_Samodurov/DICOM/PA000001/ST000000/SE000009",
    resource_folder / "01_Samodurov/DICOM/PA000001/ST000000/SE000010",
    resource_folder / "01_Samodurov/DICOM/PA000001/ST000000/SE000011",
    resource_folder / "02_Bozov/DICOM/ST00001/SE00001",
    resource_folder / "03_Burov/DICOM/ST00001/SE00001",
    resource_folder / "04_Gayvor/DICOM/PA0/ST0/SE0",
    resource_folder / "05_Golyaev/DICOM/S78510/S4010",
    resource_folder / "10_Gushchin/DICOM/19122711/26570001",
    resource_folder / "10_Gushchin/DICOM/19122711/26570002",
    resource_folder / "11_Suhushin/DICOM/PA0/ST0/SE3",
    resource_folder / "11_Suhushin/DICOM/PA0/ST0/SE4",
    resource_folder / "11_Suhushin/DICOM/PA0/ST0/SE5",
)

unrelated = (
    resource_folder / "10_Gushchin/DICOM/19122711/26570002",
    resource_folder / "11_Suhushin/DICOM/PA0/ST0/SE5",
)
related = tuple(path for path in all if path not in unrelated)

contrast = (
    resource_folder / "02_Bozov/DICOM/ST00001/SE00001",
    resource_folder / "03_Burov/DICOM/ST00001/SE00001",
    resource_folder / "04_Gayvor/DICOM/PA0/ST0/SE0",
    resource_folder / "05_Golyaev/DICOM/S78510/S4010",
    resource_folder / "10_Gushchin/DICOM/19122711/26570001",
    resource_folder / "11_Suhushin/DICOM/PA0/ST0/SE3",
    resource_folder / "11_Suhushin/DICOM/PA0/ST0/SE4",
)
uncontrast = tuple(path for path in all if path not in contrast)