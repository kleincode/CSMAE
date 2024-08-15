from src.bigearthnet_dataset.BEN_lmdb_utils import BENLMDBReader
from bigearthnet_patch_interface.patch_interface import BigEarthNetPatch
from typing import Iterator, Tuple, List
from bigearthnet_patch_interface.s1_interface import BigEarthNet_S1_Patch

class BENLMDBS1Reader(BENLMDBReader):
    """Implementation of BENLMDBReader for a LMDB database containing only BigEarthNet S1 patches.
    Adds two methods: iterate and read. Both return raw BEN S1 patches.
    """
    
    def iterate(self) -> Iterator[Tuple[str, BigEarthNet_S1_Patch]]:
        with self.env.begin(write=False) as txn:
            for key, value in txn.cursor():
                assert value is not None, f"Patch {key} unknown"
                ben_patch = BigEarthNet_S1_Patch.loads(value)
                assert isinstance(ben_patch, BigEarthNet_S1_Patch), f"Patch {key} is not an S1 patch"
                yield key, ben_patch
    
    def read(self, key) -> BigEarthNet_S1_Patch:
        bin_key = str(key).encode()
        with self.env.begin(write=False) as txn:
            binary_patch_data = txn.get(bin_key)
            assert binary_patch_data is not None, f"Patch {key} unknown"
            ben_patch = BigEarthNet_S1_Patch.loads(binary_patch_data)
            assert isinstance(ben_patch, BigEarthNet_S1_Patch), f"Patch {key} is not an S1 patch"
            return ben_patch