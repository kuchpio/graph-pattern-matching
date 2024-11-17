#include "graph6Serializer.h"

#include <sstream>
#include <cstring>

namespace core
{

graph6FormatError::graph6FormatError(const std::string& message) : std::runtime_error(message) {
}

graph6InvalidCharacterError::graph6InvalidCharacterError(std::size_t at) : graph6FormatError("Invalid character") {
    if (snprintf(_message, 32, "Invalid character at position %zd", at) < 0) strncpy(_message, graph6FormatError::what(), 32);
}

const char* graph6InvalidCharacterError::what() const noexcept {
    return _message;
}

// Format specification: https://users.cecs.anu.edu.au/~bdm/data/formats.txt
// Format parameters
constexpr unsigned char PRINT_MIN = 63, PRINT_MAX = 126, BIT_COUNT = 6, MSB_MASK = 0b100000, FULL_MASK = 0b111111;
constexpr std::string_view OPTIONAL_HEADER = ">>graph6<<";
constexpr vertex ONE_BYTE_SIZE_LIMIT = 62, THREE_BYTE_SIZE_LIMIT = 258047, SIX_BYTE_SIZE_LIMIT = 68719476735;

#define RANGE_CHECK(str, offset)                                                                                       \
    if (PRINT_MIN > str[offset] || str[offset] > PRINT_MAX) throw graph6InvalidCharacterError(offset + 1)

#define BITS_TO_CHAR(number, order) (char)(PRINT_MIN + ((number >> (order * BIT_COUNT)) & FULL_MASK))

static std::pair<vertex, std::size_t> DecodeSize(const std::string& graph6) {
    vertex size = 0;
    std::size_t offset = 0;

    if (graph6.size() == 0) throw graph6FormatError("No content");
    if (graph6.rfind(OPTIONAL_HEADER, 0) == 0) offset = OPTIONAL_HEADER.size();
    RANGE_CHECK(graph6, offset);

    if (graph6[0] == PRINT_MAX) {
        if (graph6.size() < 4) throw graph6FormatError("Invalid graph size encoding");
        RANGE_CHECK(graph6, offset + 1);
        RANGE_CHECK(graph6, offset + 2);
        RANGE_CHECK(graph6, offset + 3);

        if (graph6[1] == PRINT_MAX) {
            if (graph6.size() < 8) throw graph6FormatError("Invalid graph size encoding");
            RANGE_CHECK(graph6, offset + 4);
            RANGE_CHECK(graph6, offset + 5);
            RANGE_CHECK(graph6, offset + 6);
            RANGE_CHECK(graph6, offset + 7);

            // Six bytes
            size |= (vertex)(graph6[offset + 2] - PRINT_MIN);
            size <<= BIT_COUNT;
            size |= (vertex)(graph6[offset + 3] - PRINT_MIN);
            size <<= BIT_COUNT;
            size |= (vertex)(graph6[offset + 4] - PRINT_MIN);
            size <<= BIT_COUNT;
            size |= (vertex)(graph6[offset + 5] - PRINT_MIN);
            size <<= BIT_COUNT;
            size |= (vertex)(graph6[offset + 6] - PRINT_MIN);
            size <<= BIT_COUNT;
            size |= (vertex)(graph6[offset + 7] - PRINT_MIN);
            return std::make_pair(size, offset + 7);
        }

        // Three bytes
        size |= (vertex)(graph6[offset + 1] - PRINT_MIN);
        size <<= BIT_COUNT;
        size |= (vertex)(graph6[offset + 2] - PRINT_MIN);
        size <<= BIT_COUNT;
        size |= (vertex)(graph6[offset + 3] - PRINT_MIN);
        return std::make_pair(size, offset + 3);
    }

    // One byte
    size |= (vertex)(graph6[offset] - PRINT_MIN);
    return std::make_pair(size, offset);
}

core::Graph Graph6Serializer::Deserialize(const std::string& graph6) {

    // Decode size
    auto [size, offset] = DecodeSize(graph6);

    // Create graph
    auto graph = core::Graph(size);

    // Decode adjecency matrix
    unsigned char mask = 0;
    for (vertex u = 0; u < size; u++) {
        for (vertex v = 0; v < u; v++) {
            if (!mask) {
                mask = MSB_MASK;
                offset++;
                if (offset >= graph6.size()) throw graph6FormatError("Encoding too short");
                RANGE_CHECK(graph6, offset);
            }
            if ((graph6[offset] - PRINT_MIN) & mask) {
                graph.add_edge(u, v);
                graph.add_edge(v, u);
            }
            mask >>= 1;
        }
    }

    // Check padding
    if (offset < graph6.size() - 1) throw graph6FormatError("Encoding too long");
    while (mask) {
        if ((graph6[offset] - PRINT_MIN) & mask) throw graph6FormatError("Invalid encoding padding");
        mask >>= 1;
    }

    return graph;
}

std::string Graph6Serializer::Serialize(const core::Graph& graph) {
    std::stringstream graph6Stream;

    // Encode size
    vertex size = graph.size();
    if (size <= ONE_BYTE_SIZE_LIMIT) {
        graph6Stream << BITS_TO_CHAR(size, 0);
    } else if (size <= THREE_BYTE_SIZE_LIMIT) {
        graph6Stream << PRINT_MAX << BITS_TO_CHAR(size, 2) << BITS_TO_CHAR(size, 1) << BITS_TO_CHAR(size, 0);
    } else if (size <= SIX_BYTE_SIZE_LIMIT) {
        graph6Stream << PRINT_MAX << PRINT_MAX << BITS_TO_CHAR(size, 5) << BITS_TO_CHAR(size, 4)
                     << BITS_TO_CHAR(size, 3) << BITS_TO_CHAR(size, 2) << BITS_TO_CHAR(size, 1)
                     << BITS_TO_CHAR(size, 0);
    } else {
        throw graph6FormatError("Cannot encode graph size");
    }

    // Encode adjecency matrix
    unsigned char bitBuffer = 0, bitCtr = 0;
    for (vertex u = 0; u < graph.size(); u++) {
        for (vertex v = 0; v < u; v++) {
            bitBuffer <<= 1;
            bitCtr++;
            if (graph.has_edge(u, v)) bitBuffer |= 1;
            if (bitCtr == BIT_COUNT) {
                graph6Stream << (char)(PRINT_MIN + bitBuffer);
                bitBuffer = 0;
                bitCtr = 0;
            }
        }
    }

    // Pad with zeros
    if (bitCtr > 0) {
        bitBuffer <<= BIT_COUNT - bitCtr;
        graph6Stream << (char)(PRINT_MIN + bitBuffer);
    }

    return graph6Stream.str();
}

} // namespace core
