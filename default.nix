{ lib, stdenv, fetchFromGitHub, cmake, pkg-config, cudaPackages, gtest
, nlohmann_json }:

stdenv.mkDerivation {
  pname = "rtatblas";
  version = "unstable-2024-02-28";

  src = ./.;

  nativeBuildInputs = [ cmake pkg-config ];

  buildInputs = [ cudaPackages.cudatoolkit nlohmann_json gtest.dev ];

  meta = with lib; {
    description = "Run-time auto tuning framework for BLAS routines";
    homepage = "git@github.com:csnowdon2/rtatblas.git";
    license = licenses.mit;
    maintainers = with maintainers; [ ];
    mainProgram = "rtatblas";
    platforms = platforms.all;
  };
}
