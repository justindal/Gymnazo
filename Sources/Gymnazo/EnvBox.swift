public struct EnvBox: @unchecked Sendable {
    public let env: any Env

    public init(_ env: any Env) {
        self.env = env
    }
}
