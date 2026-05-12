#include <iostream>
#include <fstream>
#include "gguf.hpp"
#include "tokenizer.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        return 1;
    }

    std::ifstream f(argv[1], std::ios::binary);
    GGUFContext ctx;
    gguf_read_header(f, ctx.header);
    gguf_read_metadata(f, ctx);

    Tokenizer tok;
    tokenizer_init(tok, ctx);

    tokenizer_print_info(tok);

    // Check chat template
    std::cout << "\n  Chat template (primi 200 char):\n";
    if (tok.chat_template.empty()) {
        std::cout << "    (non presente)\n";
    } else {
        std::cout << "    \"" << tok.chat_template.substr(0, 200) << "...\"\n";
    }

    // Check architecture
    std::cout << "\n  Architecture: ";
    for (const auto& kv : ctx.metadata) {
        if (kv.key == "general.architecture") {
            if (auto* s = kv.value.get_if<std::string>())
                std::cout << *s;
        }
    }
    std::cout << "\n";

    // Check config
    std::cout << "\n  Config:\n";
    for (const auto& kv : ctx.metadata) {
        if (kv.key.find("qwen2.") == 0 || kv.key.find("llama.") == 0) {
            std::cout << "    " << kv.key << " = ";
            if (auto* v = kv.value.get_if<uint32_t>())
                std::cout << *v;
            else if (auto* v = kv.value.get_if<float>())
                std::cout << *v;
            else if (auto* v = kv.value.get_if<int32_t>())
                std::cout << *v;
            else
                std::cout << "?";
            std::cout << "\n";
        }
    }

    return 0;
}
