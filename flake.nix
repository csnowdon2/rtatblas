{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:numtide/nixpkgs-unfree";
    nixpkgs.inputs.nixpkgs.follows = "nixpkgs-base";
    nixpkgs-base.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { nixpkgs, ... }: {

    packages.x86_64-linux.default =
      nixpkgs.legacyPackages.x86_64-linux.callPackage (import ./default.nix)
      { };
  };
}
