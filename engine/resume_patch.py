"""Export native shuffle/duel seed state at training update boundaries."""
def apply(source, binding):
    source = '#include <sstream>\n' + source
    old = 'static std::string last_integration_response;'
    if old not in source:
        raise RuntimeError("Native resume patch context changed")
    source = source.replace(old, old + '''
static std::function<std::vector<std::string>()> export_resume_rng;
static std::function<void(const std::vector<std::string>&)> import_resume_rng;
''')
    old = '''  void reset() {
    // clock_t start = clock();'''
    if old not in source:
        raise RuntimeError("Native reset patch context changed")
    source = source.replace(old, '''  void reset() {
    export_resume_rng = [this]() {
      std::ostringstream generator, shuffle, distribution;
      generator << gen_;
      shuffle << duel_gen_;
      distribution << dist_int_;
      return std::vector<std::string>{generator.str(), shuffle.str(), distribution.str()};
    };
    import_resume_rng = [this](const std::vector<std::string>& states) {
      if (states.size() != 3) throw std::runtime_error("Invalid native RNG state");
      std::istringstream generator(states[0]), shuffle(states[1]), distribution(states[2]);
      if (!(generator >> gen_) || !(shuffle >> duel_gen_) || !(distribution >> dist_int_))
        throw std::runtime_error("Malformed native RNG state");
    };
    // clock_t start = clock();''')
    old = '  m.attr("integration_revision") = 2;'
    if old not in binding:
        raise RuntimeError("Native resume binding context changed")
    binding = binding.replace(old, old + '''
  m.attr("resume_rng_version") = 1;
  m.def("export_resume_rng", []() {
    if (!ygopro::export_resume_rng) throw std::runtime_error("Reset the engine before checkpointing");
    return ygopro::export_resume_rng();
  });
  m.def("import_resume_rng", [](const std::vector<std::string>& state) {
    if (!ygopro::import_resume_rng) throw std::runtime_error("Reset the engine before restoring");
    ygopro::import_resume_rng(state);
  });''')
    return source, binding
