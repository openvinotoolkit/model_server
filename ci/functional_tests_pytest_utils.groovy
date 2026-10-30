def validateParams(params) {
    if (!(params.TARGET_HOST ==~ /[a-zA-Z0-9_.-]+/)) {
        error "Invalid TARGET_HOST '${params.TARGET_HOST}'. Allowed characters: letters, digits, dot, underscore, hyphen."
    }
    if (!(params.TT_XDIST_WORKERS ==~ /[1-9][0-9]*/)) {
        error "Invalid TT_XDIST_WORKERS '${params.TT_XDIST_WORKERS}'. Expected a positive integer; 'auto' is not supported because tests/functional/config.py parses it as an int."
    }
}

// Splits on unquoted whitespace; '...' and "..." group words and have no escape sequences.
def splitArgs(String text) {
    def args = []
    def current = ''
    def quote = null
    def inToken = false
    for (int i = 0; i < text.length(); i++) {
        def c = text[i]
        if (quote != null) {
            if (c == quote) {
                quote = null
            } else {
                current += c
            }
        } else if (c == '"' || c == "'") {
            quote = c
            inToken = true
        } else if (c.trim().isEmpty()) {
            if (inToken) {
                args << current
                current = ''
                inToken = false
            }
        } else {
            current += c
            inToken = true
        }
    }
    if (quote != null) {
        error "Unterminated quote in PYTEST_PARAMS: ${text}"
    }
    if (inToken) {
        args << current
    }
    return args
}

def posixQuote(String arg) {
    return "'" + arg.replace("'", "'\\''") + "'"
}

// Inside cmd double quotes only %, " and a trailing backslash stay special.
def cmdQuote(String arg) {
    if (arg.contains('"') || arg.contains('%') || arg.contains('\n') || arg.contains('\r') || arg.endsWith('\\')) {
        error "Unsupported pytest argument '${arg}' on Windows: it must not contain \", %, a newline or end with a backslash."
    }
    return '"' + arg + '"'
}

// Returns PYTEST_PARAMS as a command-line fragment with every argument quoted for the target shell.
def pytestArgs(String text, boolean windows) {
    def args = splitArgs(text)
    if (args.isEmpty()) {
        error 'PYTEST_PARAMS must not be empty.'
    }
    return args.collect { arg -> windows ? cmdQuote(arg) : posixQuote(arg) }.join(' ')
}

// Returns KEY=VALUE entries suitable for withEnv(); only TT_ variables are accepted.
def parseExtraEnv(String testParameters) {
    def extraEnv = testParameters
        .readLines()
        .collect { line -> line.trim() }
        .findAll { line -> !line.isEmpty() }
    extraEnv.each { line ->
        if (!(line ==~ /TT_[A-Za-z0-9_]*=.*/)) {
            error "Invalid TEST_PARAMETERS line '${line}'. Expected TT_KEY=VALUE, one per line."
        }
    }
    return extraEnv
}

def checkoutCore(String branch, scmConfig) {
    checkout([$class: 'GitSCM', branches: [[name: "*/${branch}"]], userRemoteConfigs: [[url: scmConfig.userRemoteConfigs[0].url, credentialsId: scmConfig.userRemoteConfigs[0].credentialsId]]])
}

def publishResults() {
    junit allowEmptyResults: true, testResults: 'pytest-functional.xml'
    archiveArtifacts allowEmptyArchive: true, artifacts: 'pytest-functional.xml,test_log/**,tests/functional/test_log_build/**'
}

return this
