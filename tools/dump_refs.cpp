// Golden-dump generator for the neunote-rs port's test ladder.
//
// Built against muscriptor.cpp and run once per checkpoint size; its output is
// what the Rust port is measured against, so nothing in the port's numbers is
// trusted until it matches these.
//
// Not part of the reference: a scratch tool driving the reference's public API
// and writing the tensors the ladder compares.
//
//   g++ -std=c++23 -O2 -o build/dump_refs tools/dump_refs.cpp \
//       -Icpp/include -Icpp/src -Ibuild/_deps/ggml-src/include \
//       -Icpp/third_party/pffft build/libmuscriptor_ggml.a build/libpffft.a \
//       $(find build/_deps/ggml-build -name '*.a') -lpthread -lm -ldl
//   ./build/dump_refs <checkpoint.gguf> <fixture.wav> <out.bin> [cache-dir]
//
// Each tensor is cached as its own file under <cache-dir> and stages check it
// before doing minutes of CPU, so fixing one entry does not pay for the whole
// fixture again.

#include "muscriptor/model.hpp"
#include "muscriptor/stft.hpp"
#include "muscriptor/transcriber.hpp"
#include "trace.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace msl;

namespace
{
    constexpr int SEGMENT_SAMPLES = 80000;
    constexpr std::int32_t EOS_ID = 1;
    constexpr int CACHE_VERSION = 3;
    constexpr int KEEP_COLUMNS = 8;
    constexpr int KEEP_TAIL = 4;
    constexpr int KEEP_POSITION_ROWS = 16;

    struct Entry
    {
        std::string name;
        std::vector<std::int64_t> shape;
        std::vector<float> data;
    };

    std::vector<float> readWav(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        std::size_t at = 12;
        std::uint16_t format = 0;
        std::uint16_t channels = 0;
        std::uint32_t rate = 0;
        const char* data = nullptr;
        std::size_t data_size = 0;

        while (at + 8 <= bytes.size()) {
            char id[5] = {};
            std::memcpy(id, &bytes[at], 4);
            std::uint32_t size = 0;
            std::memcpy(&size, &bytes[at + 4], 4);

            if (std::strcmp(id, "fmt ") == 0) {
                std::memcpy(&format, &bytes[at + 8], 2);
                std::memcpy(&channels, &bytes[at + 10], 2);
                std::memcpy(&rate, &bytes[at + 12], 4);
            }
            else if (std::strcmp(id, "data") == 0) {
                data = &bytes[at + 8];
                data_size = size;
            }

            at += 8 + size + (size & 1);
        }

        if ((format != 1 && format != 3) || channels != 1 || rate != 16000 || data == nullptr) {
            std::fprintf(stderr, "expected 16 kHz mono pcm, got format=%u channels=%u rate=%u\n", format, channels, rate);
            std::exit(2);
        }

        // format 1 is pcm16, format 3 is IEEE float.
        const std::size_t width = format == 1 ? 2 : 4;
        std::vector<float> samples(data_size / width);

        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (format == 1) {
                std::int16_t value = 0;
                std::memcpy(&value, data + i * 2, 2);
                samples[i] = static_cast<float>(value) / 32768.0f;
            }
            else {
                std::memcpy(&samples[i], data + i * 4, 4);
            }
        }

        return samples;
    }

    /**
     * Keep `keep` columns along `ne[1]`, packed as `[kept, ne[0]]`.
     *
     * ggml keeps `ne[0]` contiguous, so the kept columns are `ne[0]` floats
     * apart in the source and cannot be packed into an `[ne[0], kept]` buffer.
     * They are stored the other way round instead -- one column at a time, rows
     * contiguous -- and the recorded shape says so.
     *
     * A slice that keeps a few rows and claims to have kept the columns is the
     * same bug wearing a plausible shape.
     */
    Entry sliceColumns(std::string inName, const std::vector<std::int64_t>& shape, const std::vector<float>& data,
                       std::int64_t keep, bool inFromEnd)
    {
        if (shape.size() != 2 || static_cast<std::size_t>(data.size()) != (std::size_t)(shape[0] * shape[1])) {
            return { std::move(inName), shape, data };
        }

        const std::size_t rows = static_cast<std::size_t>(shape[0]);
        const std::size_t stride = static_cast<std::size_t>(shape[1]);
        const std::size_t kept = std::min(static_cast<std::size_t>(keep), stride);
        const std::size_t start = inFromEnd ? stride - kept : 0;

        Entry entry;
        entry.name = std::move(inName);
        entry.shape = { static_cast<std::int64_t>(kept), static_cast<std::int64_t>(rows) };
        entry.data.resize(kept * rows);

        for (std::size_t column = 0; column < kept; ++column) {
            for (std::size_t row = 0; row < rows; ++row) {
                entry.data[column * rows + row] = data[row + rows * (start + column)];
            }
        }

        return entry;
    }

    /**
     * Collects tensors, computing each only when its cache file is missing.
     *
     * Stages ask `needs()` first, because the only way to get at a traced
     * intermediate is to run the graph that produces it.
     */
    class Writer
    {
    public:
        explicit Writer(std::filesystem::path cache_dir) : mCacheDir(std::move(cache_dir))
        {
            std::filesystem::create_directories(mCacheDir);
        }

        bool has(const std::string& name) const
        {
            return std::filesystem::exists(cachePath(name));
        }

        bool needs(const std::vector<std::string>& names) const
        {
            return std::any_of(names.begin(), names.end(), [&](const std::string& name) { return !has(name); });
        }

        void add(const Entry& entry)
        {
            report(entry, "computed");
            store(entry);
        }

        void keep(const Entry& entry)
        {
            if (has(entry.name)) {
                report(entry, "cached");
                return;
            }
            add(entry);
        }

        /// Write the manifest in the caller's order, so a warm and a cold run
        /// produce the same file.
        void save(const std::filesystem::path& path, const std::vector<std::string>& order) const
        {
            std::ofstream out(path, std::ios::binary);
            out.write("NNEEDMP\0", 8);

            std::vector<Entry> entries;
            for (const std::string& name : order) {
                Entry entry = load(name);
                if (entry.name.empty()) {
                    std::fprintf(stderr, "missing from the cache: %s\n", name.c_str());
                    std::exit(4);
                }
                entries.push_back(std::move(entry));
            }

            const std::uint32_t count = static_cast<std::uint32_t>(entries.size());
            out.write(reinterpret_cast<const char*>(&count), 4);

            for (const Entry& entry : entries) {
                const std::uint32_t name_len = static_cast<std::uint32_t>(entry.name.size());
                out.write(reinterpret_cast<const char*>(&name_len), 4);
                out.write(entry.name.data(), static_cast<std::streamsize>(name_len));

                const std::uint32_t n_dims = static_cast<std::uint32_t>(entry.shape.size());
                out.write(reinterpret_cast<const char*>(&n_dims), 4);
                for (const std::int64_t dim : entry.shape) {
                    out.write(reinterpret_cast<const char*>(&dim), 8);
                }

                const std::uint64_t elements = entry.data.size();
                out.write(reinterpret_cast<const char*>(&elements), 8);
                out.write(reinterpret_cast<const char*>(entry.data.data()),
                          static_cast<std::streamsize>(elements * sizeof(float)));
            }

            std::fprintf(stderr, "wrote %s (%zu tensors)\n", path.c_str(), entries.size());
        }

    private:
        static void report(const Entry& entry, const char* how)
        {
            std::fprintf(stderr, "  %-26s [", entry.name.c_str());
            for (std::size_t i = 0; i < entry.shape.size(); ++i) {
                std::fprintf(stderr, "%s%lld", i ? ", " : "", static_cast<long long>(entry.shape[i]));
            }
            std::fprintf(stderr, "]  %zu floats  %s\n", entry.data.size(), how);
        }

        std::filesystem::path cachePath(const std::string& name) const
        {
            std::string safe;
            for (const char c : name) {
                safe.push_back(std::isalnum(static_cast<unsigned char>(c)) != 0 ? c : '_');
            }
            return mCacheDir / (std::to_string(CACHE_VERSION) + "_" + safe + ".bin");
        }

        Entry load(const std::string& name) const
        {
            std::ifstream in(cachePath(name), std::ios::binary);
            if (!in) {
                return {};
            }

            std::uint32_t version = 0;
            in.read(reinterpret_cast<char*>(&version), 4);
            if (version != CACHE_VERSION) {
                return {};
            }

            Entry entry;
            entry.name = name;
            std::uint32_t n_dims = 0;
            in.read(reinterpret_cast<char*>(&n_dims), 4);
            entry.shape.resize(n_dims);
            for (std::int64_t& dim : entry.shape) {
                in.read(reinterpret_cast<char*>(&dim), 8);
            }
            std::uint64_t elements = 0;
            in.read(reinterpret_cast<char*>(&elements), 8);
            entry.data.resize(elements);
            in.read(reinterpret_cast<char*>(entry.data.data()), static_cast<std::streamsize>(elements * 4));

            return in ? entry : Entry {};
        }

        void store(const Entry& entry) const
        {
            std::ofstream out(cachePath(entry.name), std::ios::binary);
            const std::uint32_t version = CACHE_VERSION;
            out.write(reinterpret_cast<const char*>(&version), 4);
            const std::uint32_t n_dims = static_cast<std::uint32_t>(entry.shape.size());
            out.write(reinterpret_cast<const char*>(&n_dims), 4);
            for (const std::int64_t dim : entry.shape) {
                out.write(reinterpret_cast<const char*>(&dim), 8);
            }
            const std::uint64_t elements = entry.data.size();
            out.write(reinterpret_cast<const char*>(&elements), 8);
            out.write(reinterpret_cast<const char*>(entry.data.data()), static_cast<std::streamsize>(elements * 4));
        }

        std::filesystem::path mCacheDir;
    };

    std::vector<std::string> with(std::vector<std::string> names, const std::vector<std::string>& more)
    {
        names.insert(names.end(), more.begin(), more.end());
        return names;
    }

    /** Layer 0's intermediates, sliced to a few columns. */
    const std::vector<std::string> PREFILL_TRACE = {
        "pre.tok_embed", "pre.prefix",         "pre.layer_in", "blk.0.norm1",       "blk.0.attn_ctx",
        "blk.0.attn_out",                 "blk.0.res1",                     "blk.0.norm2",     "blk.0.ffn_pre_gelu",
        "blk.0.ffn_gelu", "blk.0.ffn_out",     "blk.0.out",    "post.out_norm",     "post.logits_raw",
    };

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: dump_refs <checkpoint.gguf> <fixture.wav> <out.bin> [cache-dir]\n");
        return 2;
    }

    const std::filesystem::path weights = argv[1];
    const std::filesystem::path audio = argv[2];
    const std::filesystem::path out_path = argv[3];
    const std::filesystem::path cache_dir = argc > 4 ? argv[4] : std::filesystem::path("build/refcache");

    const std::vector<float> samples = readWav(audio);
    std::fprintf(stderr, "fixture: %zu samples (%.2f s)\n", samples.size(), samples.size() / 16000.0);

    std::vector<float> chunk(static_cast<std::size_t>(SEGMENT_SAMPLES), 0.0f);
    const std::size_t available = std::min<std::size_t>(static_cast<std::size_t>(SEGMENT_SAMPLES), samples.size());
    std::copy_n(samples.begin(), available, chunk.begin());

    Writer writer(cache_dir);

    std::vector<std::string> everything = PREFILL_TRACE;
    everything = with(everything,
                      { "pre.prefix.tail",
                        "cond.mel",
                        "cond.logmel",
                        "cond.proj",
                        "cond.embed",
                        "spectrum",
                        "prefill.logits_masked",
                        "cond.dataset_name",
                        "cond.instrument_group",
                        "prefill.piano.prefix",
                        "prefill.piano.logits_masked",
                        "generate.tokens",
                        "position_table",
                        "transcribe.notes",
                        "hparams.dim" });

    const bool warm = !writer.needs(everything);
    std::fprintf(stderr, "cache is %s\n", warm ? "warm" : "cold");

    std::unique_ptr<Model> model;
    Hparams hp {};
    int n_frames = 0;
    int n_freq = 0;
    std::vector<float> conditioning;
    std::vector<std::int32_t> prefill_tokens;

    if (!warm) {
        std::fprintf(stderr, "loading %s\n", weights.c_str());
        model = std::make_unique<Model>(Model::load(weights));
        hp = model->hparams();
        n_freq = model->stft().nFreq();
        n_frames = model->stft().nFrames(SEGMENT_SAMPLES);
        prefill_tokens = { hp.initial_token_id };

        for (const auto& [name, value] :
             std::initializer_list<std::pair<const char*, float>>{
                 { "hparams.dim", static_cast<float>(hp.dim) },
                 { "hparams.n_layer", static_cast<float>(hp.n_layer) },
                 { "hparams.n_head", static_cast<float>(hp.n_head) },
                 { "hparams.head_dim", static_cast<float>(hp.head_dim) },
                 { "hparams.ffn_dim", static_cast<float>(hp.ffn_dim) },
                 { "hparams.vocab_size", static_cast<float>(hp.vocab_size) },
                 { "hparams.initial_token_id", static_cast<float>(hp.initial_token_id) },
                 { "hparams.logit_mask_start", static_cast<float>(hp.logit_mask_start) },
                 { "hparams.layer_norm_epsilon", hp.layer_norm_eps },
                 { "hparams.max_period", hp.max_period },
                 { "hparams.n_fft", static_cast<float>(hp.n_fft) },
                 { "hparams.hop_length", static_cast<float>(hp.hop_length) },
                 { "hparams.n_mels", static_cast<float>(hp.n_mels) },
                 { "hparams.log_eps", hp.log_eps },
                 { "hparams.sample_rate", static_cast<float>(hp.sample_rate) },
                 { "hparams.frame_rate", static_cast<float>(hp.frame_rate) },
             }) {
            writer.keep(Entry { name, { 1 }, { value } });
        }

        if (writer.needs({ "spectrum", "cond.mel", "cond.logmel", "cond.proj", "cond.embed" })) {
            const std::vector<float> spectrum = model->stft().magnitudes(chunk);
            writer.keep(Entry { "spectrum", { n_freq, n_frames }, spectrum });
            std::fprintf(stderr, "spectrum: %d frames x %d bins\n", n_frames, n_freq);

            Trace trace;
            conditioning = model->encodeConditioning(spectrum, n_frames, SEGMENT_SAMPLES, &trace);
            for (const char* name : { "cond.mel", "cond.logmel", "cond.proj", "cond.embed" }) {
                writer.keep(Entry { name, trace.shape(name), trace.read(name) });
            }
        }
    }

    if (model && writer.needs(with({ "prefill.logits_masked", "pre.prefix.tail", "cond.dataset_name",
                                    "cond.instrument_group" },
                                   PREFILL_TRACE))) {
        if (conditioning.empty()) {
            conditioning = model->encodeConditioning(
                model->stft().magnitudes(chunk), n_frames, SEGMENT_SAMPLES, nullptr);
        }

        Trace trace;
        const std::vector<float> logits = model->prefill(conditioning, n_frames, prefill_tokens, &trace);
        for (const std::string& name : { "cond.dataset_name", "cond.instrument_group" }) {
            if (trace.contains(name)) {
                writer.keep(Entry { name, trace.shape(name), trace.read(name) });
            }
        }
        for (const std::string& name : PREFILL_TRACE) {
            if (trace.contains(name)) {
                writer.keep(sliceColumns(name, trace.shape(name), trace.read(name), KEEP_COLUMNS, false));
            }
        }
        // The prefix's last columns: the final mel frame, then the dataset row,
        // the instrument row and the token. That order is what the port gets
        // wrong most easily.
        writer.keep(sliceColumns("pre.prefix.tail", trace.shape("pre.prefix"), trace.read("pre.prefix"), KEEP_TAIL, true));
        writer.keep(Entry { "prefill.logits_masked", { hp.vocab_size }, logits });
        std::fprintf(stderr, "prefill top-1: %d\n",
                     static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin()));
    }

    if (model && writer.needs({ "prefill.piano.logits_masked", "prefill.piano.prefix" })) {
        if (conditioning.empty()) {
            conditioning = model->encodeConditioning(
                model->stft().magnitudes(chunk), n_frames, SEGMENT_SAMPLES, nullptr);
        }

        model->reset();
        const std::vector<std::int32_t> rows { 2 };
        model->setInstrumentRows(rows);
        Trace trace;
        const std::vector<float> logits = model->prefill(conditioning, n_frames, prefill_tokens, &trace);
        if (trace.contains("pre.prefix")) {
            writer.keep(sliceColumns(
                "prefill.piano.prefix", trace.shape("pre.prefix"), trace.read("pre.prefix"), KEEP_COLUMNS, false));
        }
        writer.keep(Entry { "prefill.piano.logits_masked", { hp.vocab_size }, logits });
        model->setInstrumentRows(std::span<const std::int32_t> {});
        model->reset();
    }

    if (model && writer.needs({ "generate.tokens" })) {
        if (conditioning.empty()) {
            conditioning = model->encodeConditioning(
                model->stft().magnitudes(chunk), n_frames, SEGMENT_SAMPLES, nullptr);
        }

        const std::vector<std::int32_t> tokens =
            model->generate(conditioning, n_frames, 2000, EOS_ID, std::span<const std::int32_t> {});
        writer.keep(Entry { "generate.tokens",
                           { static_cast<std::int64_t>(tokens.size()) },
                           std::vector<float>(tokens.begin(), tokens.end()) });
        std::fprintf(stderr, "chunk 0 produced %zu tokens\n", tokens.size());
    }

    if (model && writer.needs({ "position_table" })) {
        const std::span<const float> table = model->positionEmbeddings();
        writer.keep(Entry { "position_table",
                           { KEEP_POSITION_ROWS, hp.dim },
                           std::vector<float>(table.begin(),
                                              table.begin() + static_cast<std::ptrdiff_t>(hp.dim * KEEP_POSITION_ROWS)) });
    }

    // The whole fixture, through the reference's own transcriber.
    if (writer.needs({ "transcribe.notes" })) {
        auto transcriber = Transcriber::load(weights, {});
        if (!transcriber) {
            std::fprintf(stderr, "transcriber failed to load\n");
            return 3;
        }

        TranscribeOptions options;
        options.prelude_forcing = true;
        const auto notes = transcriber->transcribe(samples, options, {});
        if (!notes) {
            std::fprintf(stderr, "transcription failed\n");
            return 3;
        }

        std::vector<float> flat;
        flat.reserve(notes->size() * 5);
        for (const Note& note : *notes) {
            flat.push_back(static_cast<float>(note.onset));
            flat.push_back(static_cast<float>(note.offset));
            flat.push_back(static_cast<float>(note.pitch));
            flat.push_back(static_cast<float>(note.program));
            flat.push_back(note.is_drum ? 1.0f : 0.0f);
        }

        std::fprintf(stderr, "fixture transcribed to %zu notes\n", notes->size());
        writer.keep(Entry { "transcribe.notes", { static_cast<std::int64_t>(notes->size()), 5 }, flat });
    }

    std::filesystem::remove(out_path);
    std::vector<std::string> order = {
        "hparams.dim",
        "hparams.n_layer",
        "hparams.n_head",
        "hparams.head_dim",
        "hparams.ffn_dim",
        "hparams.vocab_size",
        "hparams.initial_token_id",
        "hparams.logit_mask_start",
        "hparams.layer_norm_epsilon",
        "hparams.max_period",
        "hparams.n_fft",
        "hparams.hop_length",
        "hparams.n_mels",
        "hparams.log_eps",
        "hparams.sample_rate",
        "hparams.frame_rate",
    };
    order.insert(order.end(), { "spectrum", "cond.mel", "cond.logmel", "cond.proj", "cond.embed" });
    order.insert(order.end(), PREFILL_TRACE.begin(), PREFILL_TRACE.end());
    order.insert(order.end(),
                 { "pre.prefix.tail", "prefill.logits_masked", "cond.dataset_name", "cond.instrument_group",
                   "prefill.piano.prefix", "prefill.piano.logits_masked", "generate.tokens", "position_table",
                   "transcribe.notes" });
    writer.save(out_path, order);
    return 0;
}