{
  description = "xarray-sql development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Shared libraries that manylinux wheels (pyarrow, duckdb, polars,
        # numpy, ...) expect to find via the dynamic linker at runtime.
        # NixOS doesn't put these on a standard FHS path, so we point
        # LD_LIBRARY_PATH at the nix store paths that provide them.
        runtimeLibs = with pkgs; [
          stdenv.cc.cc
          zlib
          openssl
        ];

        # Pyodide pins an exact Emscripten release and Rust toolchain, both of
        # which ship as prebuilt generic-Linux binaries (emsdk's clang/node,
        # rustup toolchains). An FHS sandbox lets those binaries run unmodified
        # on NixOS. See "Building for WebAssembly" in CONTRIBUTING.md.
        #
        #   nix run .#wasm-shell                          # interactive
        #   nix run .#wasm-shell -- -c 'pyodide build'    # one-shot
        wasmShell = pkgs.buildFHSEnv {
          name = "xarray-sql-wasm";
          targetPkgs = p: with p; [
            bash
            bzip2
            cacert
            coreutils
            curl
            git
            gnumake
            gnutar
            nodejs
            python314
            rustup
            stdenv.cc
            stdenv.cc.cc.lib
            uv
            which
            xz
            zlib
            # Shared libraries for Playwright's headless Chromium, which
            # drives the browser demo in docs/demo.
            alsa-lib
            at-spi2-atk
            at-spi2-core
            atk
            cairo
            cups
            dbus
            expat
            glib
            libdrm
            libgbm
            libxkbcommon
            nspr
            nss
            pango
            systemdLibs
            xorg.libX11
            xorg.libXcomposite
            xorg.libXdamage
            xorg.libXext
            xorg.libXfixes
            xorg.libXrandr
            xorg.libxcb
          ];
          runScript = "bash";
          profile = ''
            export UV_PYTHON_PREFERENCE=only-system
            export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
          '';
        };
      in
      {
        # buildFHSEnv relies on Linux namespaces.
        packages = pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          wasm-shell = wasmShell;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            python312
            rustc
            cargo
            pkg-config
          ];

          # Use the Nix-provided interpreter above instead of having uv
          # download its own Python build.
          UV_PYTHON_PREFERENCE = "only-system";

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            echo "xarray-sql dev shell: $(python3 --version), $(rustc --version)"
          '';
        };
      });
}
