#include "graph6Serializer.h"

#include <sstream>
#include <format>

graph6FormatError::graph6FormatError(const std::string& message) 
    : std::runtime_error(message) {
}

graph6InvalidCharacterError::graph6InvalidCharacterError(std::size_t at)
	: graph6FormatError("Invalid character") {
    if (snprintf(_message, 32, "Invalid character at %zd", at) < 0) 
        strncpy(_message, graph6FormatError::what(), 32);
}

const char* graph6InvalidCharacterError::what() const {
    return _message;
}

// Format specification: https://users.cecs.anu.edu.au/~bdm/data/formats.txt 

constexpr unsigned char PRINT_MIN = 63, PRINT_MAX = 126, BIT_COUNT = 6, MSB_MASK = 0b100000;
constexpr std::string_view OPTIONAL_HEADER = ">>graph6<<";

#define RANGE_CHECK(str, offset) \
    if (PRINT_MIN > str[offset] || str[offset] > PRINT_MAX) throw graph6InvalidCharacterError(offset + 1)

std::pair<vertex, std::size_t> Graph6Serializer::DecodeSize(const std::string& graph6) {
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
            return std::make_pair(size, offset + 8);
        }

        // Three bytes
		size |= (vertex)(graph6[offset + 1] - PRINT_MIN);
		size <<= BIT_COUNT;
		size |= (vertex)(graph6[offset + 2] - PRINT_MIN);
		size <<= BIT_COUNT;
		size |= (vertex)(graph6[offset + 3] - PRINT_MIN);
        return std::make_pair(size, offset + 4);
    }
    
    // One byte
    size |= (vertex)(graph6[offset] - PRINT_MIN);
    return std::make_pair(size, offset + 1);
}

core::Graph Graph6Serializer::Deserialize(const std::string& graph6) {
    auto [ size, offset ] = DecodeSize(graph6);

    std::vector<bool> hasEdge;
    hasEdge.reserve(size * (size + 1) / 2);
    vertex i = 0;

    for (std::size_t j = offset; j < graph6.size(); j++) {
        RANGE_CHECK(graph6, j);
        unsigned char mask = MSB_MASK;
        while (mask && 2 * i < size * (size + 1)) {
            hasEdge.push_back((graph6[j] - PRINT_MIN) & mask);
            mask >>= 1;
            i++;
        }
    }

    auto graph = core::Graph(size);

    i = 0;
    for (vertex u = 0; u < size; u++) {
        for (vertex v = 0; v < u; v++) {
            if (hasEdge[i]) {
                graph.add_edge(u, v);
                graph.add_edge(v, u);
            }
            i++;
        }
    }

    return graph;
}

std::string Graph6Serializer::Serialize(const core::Graph& graph) {
    std::stringstream graph6Stream;

    if (graph.size() <= 62) {
        graph6Stream << (char)(PRINT_MIN + ((graph.size() >> 0) & 0b111111));
    } else if (graph.size() <= 258047) {
        graph6Stream << PRINT_MAX;
        graph6Stream << (char)(PRINT_MIN + ((graph.size() >> 12) & 0b111111));
        graph6Stream << (char)(PRINT_MIN + ((graph.size() >> 6) & 0b111111));
        graph6Stream << (char)(PRINT_MIN + ((graph.size() >> 0) & 0b111111));
    } else if (graph.size() <= 68719476735) {
        graph6Stream << PRINT_MAX << PRINT_MAX;
        graph6Stream << (char)(PRINT_MIN + ((graph.size() >> 30) & 0b111111));
        graph6Stream << (char)(PRINT_MIN + ((graph.size() >> 24) & 0b111111));
        graph6Stream << (char)(PRINT_MIN + ((graph.size() >> 18) & 0b111111));
        graph6Stream << (char)(PRINT_MIN + ((graph.size() >> 12) & 0b111111));
        graph6Stream << (char)(PRINT_MIN + ((graph.size() >> 6) & 0b111111));
        graph6Stream << (char)(PRINT_MIN + ((graph.size() >> 0) & 0b111111));
    } else {
        throw graph6FormatError("Cannot encode graph size");
    }

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

    if (bitCtr > 0) {
        bitBuffer <<= BIT_COUNT - bitCtr;
        graph6Stream << (char)(PRINT_MIN + bitBuffer);
    }

    return graph6Stream.str();
}
